import pandas as pd
from scipy.spatial.distance import euclidean
import numpy as np
import time
import matplotlib.pyplot as plt

dataset_true = "dataset/ISOT/True_BERT.h5"
true_training_df = pd.read_hdf(dataset_true, key='df')
#true_validation_df = pd.read_hdf(dataset, key='true_validation')
#true_test_df = pd.read_hdf(dataset, key='true_test')
#true_test_df = pd.concat([true_validation_df, true_test_df])
dataset_fake = "dataset/ISOT/Fake_BERT.h5"
fake_training_df = pd.read_hdf(dataset_fake, key='df')
#fake_validation_df = pd.read_hdf(dataset, key='fake_validation')
#fake_test_df = pd.read_hdf(dataset, key='fake_test')
#fake_test_df = pd.concat([fake_validation_df, fake_test_df])

for i in range(len(true_training_df)):

    # Extract the index 0 value of each row in the 'vector' column for true_training_df
    true_values = []
    for row in true_training_df.itertuples(index=False):
        true_values.append(row.vector[i])

    # Extract the index 0 value of each row in the 'vector' column for fake_training_df
    fake_values = []
    for row in fake_training_df.itertuples(index=False):
        fake_values.append(row.vector[i])

    # Plot the vector for true_training_df
    plt.scatter(range(len(true_values)), true_values, label='True Values', color='blue')

    # Plot the vector for fake_training_df
    plt.scatter(range(len(fake_values)), fake_values, label='Fake Values', color='red')

    plt.title('Vector at Index 0 of true_training_df and fake_training_df')
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.legend()
    # Maximize plot window
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    plt.show()
    # Combine true and fake values into a single list for plotting
    all_values = true_values + fake_values
    labels = ['True'] * len(true_values) + ['Fake'] * len(fake_values)

    