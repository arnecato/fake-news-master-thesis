import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_heatmap_of_feature(vector, feature_index=0, bins=50):
    """
    Plots a heatmap of the distribution of points for a selected feature.
    
    Parameters:
    - vector: The dataset containing feature vectors.
    - feature_index: The index of the feature to plot (default is 0 for vector[0]).
    - bins: Number of bins for the heatmap (default is 50).
    """
    # Extract the selected feature values (vector[0] in this case)
    feature_values = vector[:, feature_index]
    
    # Create a histogram for the feature values
    hist, x_edges, y_edges = np.histogram2d(np.arange(len(feature_values)), feature_values, bins=bins)
    
    cmap = plt.cm.hot
    cmap.set_under('blue')  # Set color for zero values
    plt.imshow(hist.T, origin='lower', aspect='auto', extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], cmap=cmap, vmin=0.1)

    plt.colorbar(label='Density')
    plt.xlabel('Sample Index')
    plt.ylabel(f'Feature {feature_index} Value')
    plt.title(f'Heatmap of Feature {feature_index} Distribution')
    plt.show()

# Example usage
# Assume vector is a 2D array where rows are samples and columns are feature values
df = pd.read_hdf('dataset/ISOT/True_bert.h5', key='df').sample(1000)
vector_data = np.vstack(df['vector'].values)
#vector = np.random.randn(1000, 10)  # Example dataset with 1000 samples and 10 features
plot_heatmap_of_feature(vector_data, feature_index=0, bins=100)
