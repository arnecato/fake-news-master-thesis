import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import os
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def pca(embedding):
    # Load the dataset
    dataset_path = 'dataset/ISOT/'
    fake_news = pd.read_hdf(os.path.join(dataset_path, f'Fake_{embedding}.h5'))
    real_news = pd.read_hdf(os.path.join(dataset_path, f'True_{embedding}.h5'))

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
    X = np.array(df['vector'].tolist())

    print(f"Vector shape: {X.shape}")  # Should be (n_samples, 768)

    # Apply PCA
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
    plt.title(f'PCA of News Dataset {embedding}')
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%})')
    plt.legend()
    plt.grid(True)
    plt.savefig('report/results/pca_results_{embedding}.png')
    plt.show()

    # Print variance explained by each component
    print("Explained variance ratio:")
    print(pca.explained_variance_ratio_)
    print("Total variance explained:", sum(pca.explained_variance_ratio_))

embeddings = ['roberta-base', 'fasttext', 'distilbert-base-cased', 'bert-base-cased']
for embedding in embeddings:
    pca(embedding)
    print(f"Finished PCA for {embedding}")
    print("===================================")