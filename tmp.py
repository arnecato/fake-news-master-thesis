import pandas as pd
import h5py

import matplotlib.pyplot as plt

# Load the dataset
file_path = 'dataset/ISOT/True_BERT.h5'

# Convert to DataFrame
df = pd.read_hdf(file_path, key='df')

print(df.head())