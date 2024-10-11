import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import cdist


def plot_heatmap_of_feature(vectors_A, vectors_B, num_bins):
    # Convert the list to a NumPy array
    data_array_A = np.array(vectors_A)
    data_array_B = np.array(vectors_B)
    
    # Define min and max values dynamically
    combined_data = np.vstack((data_array_A, data_array_B))  # To calculate common bin range
    min_value = combined_data.min()
    max_value = combined_data.max()

    # Calculate bin size based on the number of bins
    bins = np.linspace(min_value, max_value, num_bins + 1)

    # Bin the data separately for A and B
    binned_data_A = np.digitize(data_array_A, bins) - 1  # Subtract 1 to align bin indexes with Python's 0-indexing
    binned_data_B = np.digitize(data_array_B, bins) - 1

    # Create frequency matrices for A and B
    num_features_A = data_array_A.shape[1]
    num_features_B = data_array_B.shape[1]
    
    frequency_matrix_A = np.zeros((len(bins), num_features_A))
    frequency_matrix_B = np.zeros((len(bins), num_features_B))

    # Fill in the frequency matrix for A
    for i in range(num_features_A):
        for j in range(len(bins)):
            frequency_matrix_A[j, i] = np.sum(binned_data_A[:, i] == j)

    # Fill in the frequency matrix for B
    for i in range(num_features_B):
        for j in range(len(bins)):
            frequency_matrix_B[j, i] = np.sum(binned_data_B[:, i] == j)

    # Interleave frequency matrices of A and B
    interleaved_matrix = np.zeros((len(bins), num_features_A + num_features_B))
    for i in range(num_features_A):
        interleaved_matrix[:, 2 * i] = frequency_matrix_A[:, i]  # Feature from A
        interleaved_matrix[:, 2 * i + 1] = frequency_matrix_B[:, i]  # Corresponding feature from B

    # Plot the heatmap
    plt.figure(figsize=(16, 6))  # Adjust the figure size
    ax = sns.heatmap(interleaved_matrix, annot=True, cmap="YlGnBu", cbar=True,
                     yticklabels=np.round(bins, 2), fmt='.0f',
                     annot_kws={"size": 8, "weight": "bold", "color": "black"})
    
    ax.set_aspect(aspect='auto')
    
    # Set labels for the columns
    feature_labels = []
    for i in range(num_features_A):
        feature_labels.append(f'Feature {i+1} (A)')
        feature_labels.append(f'Feature {i+1} (B)')
    
    ax.set_xticklabels(feature_labels, rotation=45, ha="right", fontsize=10)
    
    # Set axis labels
    plt.xlabel("Feature Position")
    plt.ylabel("Binned Feature Values")
    plt.title(f"Heatmap of Feature Values Across {num_bins} Bins for DataFrame A and DataFrame B")

    # Invert the y-axis to increase upwards
    plt.gca().invert_yaxis()

    # Show the plot
    plt.show()

def extract_and_plot_two_features(df_true, df_fake, feature1_index, feature2_index):
    """
    Extracts two features from True and Fake datasets and plots their distribution in a 2D scatter plot.

    Parameters:
    df_true (pd.DataFrame): DataFrame containing true samples.
    df_fake (pd.DataFrame): DataFrame containing fake samples.
    feature1_index (int): Index of the first feature to extract.
    feature2_index (int): Index of the second feature to extract.
    """
    # Extract feature 1 and feature 2 from True dataset
    feature1_true = np.vstack(df_true['vector'])[:, feature1_index]
    feature2_true = np.vstack(df_true['vector'])[:, feature2_index]
    
    # Extract feature 1 and feature 2 from Fake dataset
    feature1_fake = np.vstack(df_fake['vector'])[:, feature1_index]
    feature2_fake = np.vstack(df_fake['vector'])[:, feature2_index]

    # Plot the distribution in 2D scatter plot
    plt.figure(figsize=(10, 6))
    
    plt.scatter(feature1_true, feature2_true, label='True', color='blue', alpha=0.25)
    plt.scatter(feature1_fake, feature2_fake, label='Fake', color='red', alpha=0.25)

    # Set labels and legend
    plt.xlabel(f'Feature {feature1_index + 1}')
    plt.ylabel(f'Feature {feature2_index + 1}')
    plt.title('2D Feature Distribution for True and Fake Samples')
    plt.legend()

    # Show the plot
    plt.show()

def measure_average_distance(df_true, df_fake, feature_indices):
    """
    Measures the average distance between true and fake samples based on selected features.

    Parameters:
    df_true (pd.DataFrame): DataFrame containing true samples.
    df_fake (pd.DataFrame): DataFrame containing fake samples.
    feature_indices (list of int): List of feature indices to be used for distance measurement.
    """
    # Extract selected features from both datasets
    vectors_true = np.vstack(df_true['vector'])[:, feature_indices]
    vectors_fake = np.vstack(df_fake['vector'])[:, feature_indices]

    # Calculate pairwise distances between true and fake samples
    distances = cdist(vectors_true, vectors_fake, metric='euclidean')

    # Calculate the average distance
    average_distance = np.mean(distances)
    return average_distance

def main():
    # Example usage
    # Load your two dataframes (df_A and df_B) with vector data from two sources
    #df_A = pd.read_hdf('dataset/ISOT/TRUE_bert.h5', key='df') #.sample(1000)
    #df_B = pd.read_hdf('dataset/ISOT/FAKE_bert.h5', key='df') #.sample(1000)
    df_A = pd.read_csv('dataset/ISOT/True_Fake.csv') #.sample(1000)
    #df_B = pd.read_csv('dataset/ISOT/Fake.csv') #.sample(1000)
    # Filter dataframes to include only those with feature index 0 values between -1 and 0
    min_length_A = df_A['text'].str.len().min()
    max_length_A = df_A['text'].str.len().max()
    avg_length_A = df_A['text'].str.len().mean()
    median_length_A = df_A['text'].str.len().median()

    print(f"Median text length in df_A: {median_length_A}")
    print(f"Average text length in df_A: {avg_length_A}")
    print(f"Minimum length in df_A: {min_length_A}, Maximum length in df_A: {max_length_A}")
    for _ in range(100):
        feature1_index = np.random.randint(0, 768)
        feature2_index = np.random.randint(0, 768)
        avg_dist = measure_average_distance(df_A, df_B, [feature1_index, feature2_index])
        print(avg_dist)
        if avg_dist > 0.25:
            extract_and_plot_two_features(df_A, df_B, feature1_index=feature1_index, feature2_index=feature2_index)

# 404 and 99 feature value, 594/558, 
    '''for _ in range(100):
        random_value = np.random.uniform(-0.5, 0.5)
        print(random_value)
        df_A_tmp = df_A[(np.vstack(df_A['vector'])[:, 0] > random_value)]
        df_B_tmp = df_B[(np.vstack(df_B['vector'])[:, 0] > random_value)]
        average_distance = measure_average_distance(df_A_tmp, df_B_tmp, [2, 3])
        print(f'Random value: {random_value}, Average distance: {average_distance}, Number of samples in df_A: {len(df_A_tmp)}, Number of samples in df_B: {len(df_B_tmp)}')
        if average_distance > 0.2:
            extract_and_plot_two_features(df_A_tmp, df_B_tmp, feature1_index=2, feature2_index=3)'''
    # Measure average distance between true and fake samples based on selected features
    #average_distance = measure_average_distance(df_A.sample(100), df_B.sample(100), feature_indices)
    #print(f'Average distance between True and Fake samples (based on features {feature_indices}): {average_distance}')
    #if average_distance > 0.3:
    #    extract_and_plot_two_features(df_A, df_B, feature1_index=feature_indices[0], feature2_index=feature_indices[1])'''        
    

if __name__ == "__main__":
    main()