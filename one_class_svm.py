import pandas as pd
from scipy.spatial.distance import euclidean
import numpy as np
import time
from sklearn.svm import OneClassSVM
dataset = "dataset/ISOT/True_Fake_bert_umap_3dim_3600_6000.h5"
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

# Fit the OneClassSVM model
ocsvm = OneClassSVM(kernel='rbf', gamma='auto').fit(X_train)


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

print('True positive:', true_positive, 'False negative:', false_negative, 'False positive:', false_positive, 'True negative:', true_negative, 'Precision tp/(tp+fp)', true_positive/(true_positive+false_positive), 'Recall tp/(tp+fn):', true_positive/(true_positive+false_negative))