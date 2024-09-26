import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_heatmap_of_feature(vectors, num_bins):
    # Example list of vectors with float values
    #data = [[0.2, 1.1, 3.9, 2.2], [2.1, 3.3, 4.0, 1.1], [1.0, 2.2, 3.8, 0.5]]  # Add more vectors as needed

    # Convert the list to a NumPy array
    data_array = np.array(vectors)

    # Define min and max values dynamically
    min_value = data_array.min()
    max_value = data_array.max()

    # Number of bins (you can specify this as desired)
    #num_bins = 10  # For example, 10 bins

    # Calculate bin size based on the number of bins
    bin_size = (max_value - min_value) / num_bins

    # Create bins based on the number of bins
    bins = np.linspace(min_value, max_value, num_bins + 1)

    # Bin the data for each vector
    binned_data = np.digitize(data_array, bins) - 1  # Subtract 1 to align bin indexes with Python's 0-indexing

    # Create a frequency matrix for the binned data
    num_features = data_array.shape[1]
    frequency_matrix = np.zeros((len(bins), num_features))

    # Fill in the frequency matrix
    for i in range(num_features):
        for j in range(len(bins)):
            frequency_matrix[j, i] = np.sum(binned_data[:, i] == j)

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(frequency_matrix, annot=True, cmap="YlGnBu", cbar=True, yticklabels=np.round(bins, 2))

    # Set labels
    plt.xlabel("Feature Position (x-axis)")
    plt.ylabel("Binned Feature Values (y-axis)")
    plt.title(f"Heatmap of Feature Values Across {num_bins} Bins")

    # Invert the y-axis to increase upwards
    plt.gca().invert_yaxis()

    # Show the plot
    plt.show()

# Example usage
# Assume vector is a 2D array where rows are samples and columns are feature values
df = pd.read_hdf('dataset/ISOT/FAKE_bert.h5', key='df').sample(10000)
vector_data = np.vstack(df['vector'])
vector_data = vector_data[:, :10]  # Select the first 10 features for visualization
#vector = np.random.randn(1000, 10)  # Example dataset with 1000 samples and 10 features
plot_heatmap_of_feature(vector_data, 50)
