import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import os
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def pca(embedding, matrix=False, batch_size=1000):
    # Load the dataset
    dataset_path = 'dataset/ISOT/'
    fake_news = pd.read_hdf(os.path.join(dataset_path, f'Fake_{embedding}.h5')) #.sample(10000, random_state=42)
    real_news = pd.read_hdf(os.path.join(dataset_path, f'True_{embedding}.h5')) #.sample(10000, random_state=42)

    # Add a label column
    fake_news['label'] = 0  # 0 for fake
    real_news['label'] = 1  # 1 for real

    # Concatenate datasets
    df = pd.concat([fake_news, real_news], axis=0)
    df = df.reset_index(drop=True)

    # Basic preprocessing - drop NaN values
    df = df.dropna()

    # Extract the vector values (embeddings)
    # Convert the vector column from lists to a numpy array
    X = np.array(df['vector'].tolist(), dtype=np.float64)

    # Apply dimensionality reduction in batches for the matrix case
    if matrix:
        print("Creating matrix of pairwise products in batches...")
        n_samples, n_features = X.shape
        
        # Initialize PCA
        pca = PCA(n_components=2)
        
        # Process in batches to reduce memory usage
        all_pca_results = []
        
        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            print(f"Processing batch {i//batch_size + 1}/{(n_samples + batch_size - 1)//batch_size}: samples {i} to {batch_end-1}")
            
            # Get the current batch
            X_batch = X[i:batch_end]
            
            # Process each sample in the batch
            X_batch_matrix = np.zeros((batch_end - i, n_features * n_features), dtype=np.float16)
            
            for j in range(len(X_batch)):
                vec = X_batch[j]
                # Compute outer product (all pairwise multiplications)
                outer_prod = np.outer(vec, vec).flatten()
                # Store the flattened result
                X_batch_matrix[j] = outer_prod
            
            # Either fit_transform or transform depending on whether this is the first batch
            if i == 0:
                batch_pca_result = pca.fit_transform(X_batch_matrix)
            else:
                batch_pca_result = pca.transform(X_batch_matrix)
            
            all_pca_results.append(batch_pca_result)
        
        # Combine results
        X_pca = np.vstack(all_pca_results)
        print(f"Final PCA shape: {X_pca.shape}")
    else:
        print(f"Vector shape: {X.shape}")  # Should be (n_samples, 768)
        # Apply PCA directly when not using matrix transformation
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

    # Create a DataFrame with the principal components
    pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
    pca_df['label'] = df['label'].values

    # Plot the results
    plt.figure(figsize=(10, 8))
    plt.scatter(pca_df[pca_df['label']==0]['PC1'], pca_df[pca_df['label']==0]['PC2'], 
                c='red', label='Fake News', alpha=0.5)
    plt.scatter(pca_df[pca_df['label']==1]['PC1'], pca_df[pca_df['label']==1]['PC2'], 
                c='blue', label='Real News', alpha=0.5)
    plt.title(f'PCA of News Dataset {embedding}' + (' with Matrix Transform' if matrix else ''))
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'report/results/pca_results_{embedding}{"_matrix" if matrix else ""}.png')
    plt.show()

    # Print variance explained by each component
    print("Explained variance ratio:")
    print(pca.explained_variance_ratio_)
    print("Total variance explained:", sum(pca.explained_variance_ratio_))

embeddings = ['roberta-base', 'fasttext', 'distilbert-base-cased', 'bert-base-cased']
for embedding in embeddings:
    pca(embedding, True)
    print(f"Finished PCA for {embedding}")
    print("===================================")