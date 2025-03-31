import pandas as pd
from scipy.spatial.distance import euclidean
import numpy as np
import time
from sklearn.svm import OneClassSVM
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import f1_score

# Hyperparameters tuned to increase recall
nu = 0.01  # Decreased to be more inclusive of the target class
gamma = 0.005  # Lower gamma makes decision boundary smoother
kernel = 'rbf'  # RBF kernel works well for this task
tol = 1e-4  # Lower tolerance for more precise convergence
shrinking = True  # Keep shrinking heuristic for efficiency
cache_size = 500  # Increased cache size for better performance

dataset = "dataset/ISOT/True_Fake_roberta-base_umap_1dim_15_25700_21417.h5"
true_training_df = pd.read_hdf(dataset, key='true_training')
true_validation_df = pd.read_hdf(dataset, key='true_validation')
true_test_df = pd.read_hdf(dataset, key='true_test')
true_test_df = pd.concat([true_validation_df, true_test_df])

fake_training_df = pd.read_hdf(dataset, key='fake_training')
fake_validation_df = pd.read_hdf(dataset, key='fake_validation')
fake_test_df = pd.read_hdf(dataset, key='fake_test')
fake_test_df = pd.concat([fake_validation_df, fake_test_df])

# Ensure vectors are stacked correctly
X_train = np.vstack([np.vstack(true_training_df['vector'].values)])

# Fit the OneClassSVM model with exposed hyperparameters
ocsvm = OneClassSVM(
    kernel=kernel, 
    gamma=gamma, 
    nu=nu, 
    tol=tol, 
    shrinking=shrinking,
    cache_size=cache_size
).fit(X_train)

# Prepare validation data
true_val_vectors = np.vstack(true_validation_df['vector'].values)
fake_val_vectors = np.vstack(fake_validation_df['vector'].values)

# Grid search parameters
param_grid = {
    'nu': [0.05, 0.1],
    'gamma': [0.001, 0.005, 0.01, 0.05],
    'kernel': ['rbf', 'sigmoid', 'poly'],
    'tol': [1e-3, 5e-4]
}

grid = ParameterGrid(param_grid)
best_f1 = 0
best_params = None
patience = 5
no_improvement = 0
history = []

print("Starting grid search...")
start_time = time.time()

for params in grid:
    # Train model with current parameters
    print(f'Fitting... {params}')
    model = OneClassSVM(
        kernel=params['kernel'],
        gamma=params['gamma'],
        nu=params['nu'],
        tol=params['tol'],
        shrinking=shrinking,
        cache_size=cache_size
    ).fit(X_train)
    print('Predicting...')
    # Predict on validation set
    true_val_pred = model.predict(true_val_vectors)
    fake_val_pred = model.predict(fake_val_vectors)
    
    # Create labels (1 for true news, -1 for fake news)
    y_true = np.concatenate([np.ones(len(true_val_vectors)), -1 * np.ones(len(fake_val_vectors))])
    y_pred = np.concatenate([true_val_pred, fake_val_pred])
    
    # Calculate F1 score
    current_f1 = f1_score(y_true, y_pred, pos_label=1)
    
    # Store result
    result = {
        'params': params,
        'f1_score': current_f1,
        'model': model
    }
    history.append(result)
    
    print(f"Parameters: {params}, F1 Score: {current_f1:.4f}")
    
    # Check if we have improvement
    if current_f1 > best_f1:
        best_f1 = current_f1
        best_params = params
        best_model = model
        no_improvement = 0
    else:
        no_improvement += 1

print(f"\nGrid search completed in {time.time() - start_time:.2f} seconds")
print(f"Best parameters: {best_params}")
print(f"Best validation F1 score: {best_f1:.4f}")

# Update our model to the best one found
ocsvm = best_model

# Sorting results to see trends
history.sort(key=lambda x: x['f1_score'], reverse=True)
print("\nTop 5 parameter combinations:")
for i in range(min(5, len(history))):
    print(f"F1: {history[i]['f1_score']:.4f}, Params: {history[i]['params']}")

# Predict on the test data
true_test_vectors = np.vstack(true_test_df['vector'].values)
fake_test_vectors = np.vstack(fake_test_df['vector'].values)

true_test_predictions = ocsvm.predict(true_test_vectors)
fake_test_predictions = ocsvm.predict(fake_test_vectors)

# Calculate metrics
true_positive = np.sum(true_test_predictions == 1)
false_negative = np.sum(true_test_predictions == -1)
false_positive = np.sum(fake_test_predictions == 1)
true_negative = np.sum(fake_test_predictions == -1)

print(f'True positive: {true_positive}, False negative: {false_negative}, False positive: {false_positive}, True negative: {true_negative}')
print(f'Precision tp/(tp+fp): {true_positive/(true_positive+false_positive):.4f}, Recall tp/(tp+fn): {true_positive/(true_positive+false_negative):.4f}')

