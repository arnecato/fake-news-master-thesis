import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import umap

import matplotlib.pyplot as plt


dim = 2
neighbors = 900
min_dist = 0.1
word_embedding = 'bert' # 'glove' or 'bert'
#vec_factory = BERTVectorFactory() # SpacyVectorFactory() # BERTVectorFactory() # SpacyVectorFactory() #BERTVectorFactory()
true_df = pd.read_hdf(f'dataset/ISOT/True_{word_embedding}.h5', key='df') 
# center vectors
#umap_embeddings_true = np.vstack(true_df['vector'].values)
#mean_true = umap_embeddings_true.mean(axis=0)
#umap_embeddings_true_centered = umap_embeddings_true - mean_true
#true_df['vector'] = [np.array(vec) for vec in umap_embeddings_true_centered] # TODO: move this code to dataset processing. UMAP, centering etc.
#true_df = vec_factory.load_vectorized_dataframe('dataset/ISOT/True_glove.h5')
true_df = true_df.sample(1000, random_state=42)
dimension_reducer = umap.UMAP(n_components=dim, n_neighbors=neighbors, random_state=42, min_dist=min_dist) #PCA(n_components=dim)
#umap_reducer = umap.UMAP(n_components=dim, n_jobs=-1, n_neighbors=neighbors)
vectors_df = pd.DataFrame(true_df['vector'].tolist())
reduced_vectors = dimension_reducer.fit_transform(vectors_df)

true_df['vector'] = [np.array(vec) for vec in reduced_vectors]
true_df.to_hdf(f'dataset/ISOT/True_{word_embedding}_umap_{dim}dim_{neighbors}_{min_dist}.h5', key='df', mode='w')
print('Dim reduced to:', len(true_df.iloc[0]['vector']))
true_training_df, tmp_df = train_test_split(true_df, test_size=0.4, random_state=42)
true_validation_df, true_test_df = train_test_split(tmp_df, test_size=0.5, random_state=42)

fake_df = pd.read_hdf(f'dataset/ISOT/Fake_{word_embedding}.h5', key='df')
# center fake vectors based on true mean
#umap_embeddings_fake = np.vstack(fake_df['vector'].values)
#umap_embeddings_fake_centered = umap_embeddings_fake - mean_true
#fake_df['vector'] = [np.array(vec) for vec in umap_embeddings_fake_centered]

fake_df = fake_df.sample(1000, random_state=42)
vectors_df = pd.DataFrame(fake_df['vector'].tolist())
reduced_vectors = dimension_reducer.transform(vectors_df)
fake_df['vector'] = [np.array(vec) for vec in reduced_vectors]
fake_df.to_hdf(f'dataset/ISOT/Fake_{word_embedding}_umap_{dim}dim_{neighbors}_{min_dist}.h5', key='df', mode='w')
fake_training_df, tmp_df = train_test_split(fake_df, test_size=0.4, random_state=42)
fake_validation_df, fake_test_df = train_test_split(tmp_df, test_size=0.5, random_state=42)


# Extract the 2D vectors
true_vectors_2d = np.vstack(true_df['vector'].values)
true_validation_df = pd.concat([true_validation_df, true_test_df])
true_validation_df_vectors_2d = np.vstack(true_validation_df['vector'].values)
fake_vectors_2d = np.vstack(fake_df['vector'].values)
fake_validation_df = pd.concat([fake_validation_df, fake_test_df])
fake_validation_df_vectors_2d = np.vstack(fake_validation_df['vector'].values)
                                        

# Plot the 2D vectors
plt.figure(figsize=(10, 8))
plt.scatter(true_vectors_2d[:, 0], true_vectors_2d[:, 1], s=5, color='blue', alpha=0.25)
plt.scatter(fake_vectors_2d[:, 0], fake_vectors_2d[:, 1], s=5, color='red', alpha=0.25)
plt.scatter(true_validation_df_vectors_2d[:, 0], true_validation_df_vectors_2d[:, 1], s=5, color='blue', alpha=0.25)
plt.scatter(fake_validation_df_vectors_2d[:, 0], fake_validation_df_vectors_2d[:, 1], s=5, color='red', alpha=0.25)
plt.title(f'UMAP {word_embedding} - {neighbors} neighbors - {min_dist} min_dist')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.show()
