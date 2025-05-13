import pandas as pd
import numpy as np
import time
from sklearn.svm import SVC
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import f1_score, precision_score, recall_score

dataset = "dataset/ISOT/True_Fake_roberta-base_umap_1dim_15_25700_21417.h5"
true_training_df = pd.read_hdf(dataset, key='true_training')
true_validation_df = pd.read_hdf(dataset, key='true_validation')
true_test_df = pd.read_hdf(dataset, key='true_test')
true_test_df = pd.concat([true_validation_df, true_test_df])

fake_training_df = pd.read_hdf(dataset, key='fake_training')
fake_validation_df = pd.read_hdf(dataset, key='fake_validation')
fake_test_df = pd.read_hdf(dataset, key='fake_test')
fake_test_df = pd.concat([fake_validation_df, fake_test_df])

# Prepare training data with both classes and labels
X_train_true = np.vstack(true_training_df['vector'].values)
X_train_fake = np.vstack(fake_training_df['vector'].values)
X_train = np.vstack([X_train_true, X_train_fake])
y_train = np.concatenate([np.ones(len(X_train_true)), np.zeros(len(X_train_fake))])

# Initial SVM model with default parameters
svm = SVC(
    kernel='rbf',
    gamma='scale',
    C=1.0,
    class_weight='balanced',
    probability=True
).fit(X_train, y_train)

# Prepare validation data
true_val_vectors = np.vstack(true_validation_df['vector'].values)
fake_val_vectors = np.vstack(fake_validation_df['vector'].values)
X_val = np.vstack([true_val_vectors, fake_val_vectors])
y_val = np.concatenate([np.ones(len(true_val_vectors)), np.zeros(len(fake_val_vectors))])

# Grid search parameters
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'kernel': ['rbf', 'sigmoid', 'poly'],
    'class_weight': ['balanced', None]
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
    model = SVC(
        kernel=params['kernel'],
        gamma=params['gamma'],
        C=params['C'],
        class_weight=params['class_weight'],
        probability=True
    ).fit(X_train, y_train)
    
    print('Predicting...')
    # Predict on validation set
    y_pred = model.predict(X_val)
    
    # Calculate F1 score
    current_f1 = f1_score(y_val, y_pred, pos_label=1)
    
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
    
    # Early stopping
    if no_improvement >= patience:
        print("No improvement for several iterations. Stopping early.")
        break

print(f"\nGrid search completed in {time.time() - start_time:.2f} seconds")
print(f"Best parameters: {best_params}")
print(f"Best validation F1 score: {best_f1:.4f}")

# Update our model to the best one found
svm = best_model

# Sorting results to see trends
history.sort(key=lambda x: x['f1_score'], reverse=True)
print("\nTop 5 parameter combinations:")
for i in range(min(5, len(history))):
    print(f"F1: {history[i]['f1_score']:.4f}, Params: {history[i]['params']}")

# Predict on the test data
true_test_vectors = np.vstack(true_test_df['vector'].values)
fake_test_vectors = np.vstack(fake_test_df['vector'].values)
X_test = np.vstack([true_test_vectors, fake_test_vectors])
y_test = np.concatenate([np.ones(len(true_test_vectors)), np.zeros(len(fake_test_vectors))])

y_test_pred = svm.predict(X_test)

# Calculate metrics
true_positive = np.sum((y_test == 1) & (y_test_pred == 1))
false_negative = np.sum((y_test == 1) & (y_test_pred == 0))
false_positive = np.sum((y_test == 0) & (y_test_pred == 1))
true_negative = np.sum((y_test == 0) & (y_test_pred == 0))

print(f'True positive: {true_positive}, False negative: {false_negative}, False positive: {false_positive}, True negative: {true_negative}')
precision = precision_score(y_test, y_test_pred, pos_label=1)
recall = recall_score(y_test, y_test_pred, pos_label=1)
print(f'Precision: {precision:.4f}, Recall: {recall:.4f}')

# Calculate F1-score for test set
test_f1 = f1_score(y_test, y_test_pred, pos_label=1)
print(f'F1-score for test set: {test_f1:.4f}')
