import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import umap
import argparse
import matplotlib.pyplot as plt
import time
import h5py
import warnings
warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)

def reduce_dimensions(filepath_true, filepath_fake, dim, neighbors, word_embedding, sample_size=1000):
    time0 = time.perf_counter()
    true_df = pd.read_hdf(filepath_true, key='df') 
    # center vectors
    #umap_embeddings_true = np.vstack(true_df['vector'].values)
    #mean_true = umap_embeddings_true.mean(axis=0)
    #umap_embeddings_true_centered = umap_embeddings_true - mean_true
    #true_df['vector'] = [np.array(vec) for vec in umap_embeddings_true_centered] # TODO: move this code to dataset processing. UMAP, centering etc.
    if sample_size == -1:
        sample_size = len(true_df)
    true_df = true_df.sample(sample_size, random_state=42)
    dimension_reducer = umap.UMAP(n_components=dim, n_neighbors=neighbors, n_jobs=-1) 
    vectors_df = pd.DataFrame(true_df['vector'].tolist())
    reduced_vectors = dimension_reducer.fit_transform(vectors_df)
    true_df['vector'] = [np.array(vec) for vec in reduced_vectors]
    true_training_df, tmp_df = train_test_split(true_df, test_size=0.6, random_state=42)
    true_validation_df, true_test_df = train_test_split(tmp_df, test_size=0.5, random_state=42)
    
    fake_df = pd.read_hdf(filepath_fake, key='df')
    fake_df = fake_df.sample(sample_size, random_state=42)
    vectors_df = pd.DataFrame(fake_df['vector'].tolist())
    reduced_vectors = dimension_reducer.transform(vectors_df)
    fake_df['vector'] = [np.array(vec) for vec in reduced_vectors]
    fake_training_df, tmp_df = train_test_split(fake_df, test_size=0.6, random_state=42)
    fake_validation_df, fake_test_df = train_test_split(tmp_df, test_size=0.5, random_state=42)
    
    # save to file
    if sample_size == -1:
        sample_size = 'all'
    filepath = f'dataset/ISOT/True_Fake_{word_embedding}_umap_{dim}dim_{neighbors}_{sample_size}.h5'
    true_training_df.to_hdf(filepath, key='true_training', mode='a')
    true_validation_df.to_hdf(filepath, key='true_validation', mode='a')
    true_test_df.to_hdf(filepath, key='true_test', mode='a')
    fake_training_df.to_hdf(filepath, key='fake_training', mode='a')
    fake_validation_df.to_hdf(filepath, key='fake_validation', mode='a')
    fake_test_df.to_hdf(filepath, key='fake_test', mode='a')
    # add metadata
    with h5py.File(filepath, 'a') as f:
        # Add metadata at the file level
        f.attrs['description'] = 'ISOT Dataset with True and Fake news'
        f.attrs['word_embedding'] = word_embedding
        f.attrs['dim'] = dim
        f.attrs['neighbors'] = neighbors
        f.attrs['sample_size'] = sample_size

    print('Dim reduced to:', len(true_df.iloc[0]['vector']), 'File', filepath, 'Processing time:', time.perf_counter()-time0)

def plot_file(filepath, true_keys, fake_keys):
    true_df = pd.read_hdf(filepath, key=true_keys[0])
    for key in true_keys[1:]:
        true_df = pd.concat([true_df, pd.read_hdf(filepath, key=key)])
    fake_df = pd.read_hdf(filepath, key=fake_keys[0])
    for key in fake_keys[1:]:
        fake_df = pd.concat([fake_df, pd.read_hdf(filepath, key=key)])
    
    # read metadata from hdf5 file
    with h5py.File(filepath, 'r') as f:
        word_embedding = f.attrs['word_embedding']
        dim = f.attrs['dim']
        neighbors = f.attrs['neighbors']
        sample_size = f.attrs['sample_size']
    plot(true_df, fake_df, word_embedding, neighbors, sample_size)

def plot(true_df, fake_df, word_embedding, neighbors, sample_size):
    # Extract the 2D vectors
    true_vectors_2d = np.vstack(true_df['vector'].values)
    #true_validation_df = pd.concat([true_validation_df, true_test_df])
    #true_validation_df_vectors_2d = np.vstack(true_validation_df['vector'].values)
    fake_vectors_2d = np.vstack(fake_df['vector'].values)
    #fake_validation_df = pd.concat([fake_validation_df, fake_test_df])
    #fake_validation_df_vectors_2d = np.vstack(fake_validation_df['vector'].values)
                                            

    # Plot the 2D vectors
    plt.figure(figsize=(10, 8))
    plt.scatter(true_vectors_2d[:, 0], true_vectors_2d[:, 1], s=5, color='blue', alpha=0.25)
    plt.scatter(fake_vectors_2d[:, 0], fake_vectors_2d[:, 1], s=5, color='red', alpha=0.25)
    #plt.scatter(true_validation_df_vectors_2d[:, 0], true_validation_df_vectors_2d[:, 1], s=5, color='blue', alpha=0.25)
    #plt.scatter(fake_validation_df_vectors_2d[:, 0], fake_validation_df_vectors_2d[:, 1], s=5, color='red', alpha=0.25)
    plt.title(f'UMAP {word_embedding} - {neighbors} neighbors - {sample_size} samples')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='UMAP Dimensionality Reduction and Other Options')
    subparsers = parser.add_subparsers(dest='command', help='Sub-command to run')
   
    umap_parser = subparsers.add_parser('umap', help='Perform UMAP dimensionality reduction')
    umap_parser.add_argument('--filepath_true', type=str, required=True, help='Path to the input HDF5 file for true news')
    umap_parser.add_argument('--filepath_fake', type=str, required=True, help='Path to the input HDF5 file for fake news')
    umap_parser.add_argument('--dim', type=int, required=True, help='Number of dimensions for UMAP')
    umap_parser.add_argument('--word_embedding', type=str, required=True, choices=['glove', 'bert'], help='Type of word embedding')
    umap_parser.add_argument('--neighbors', type=int, default=15, help='Number of neighbors for UMAP')
    umap_parser.add_argument('--sample_size', type=int, default=-1, help='Sample size for the dataset')
    
    plot_parser = subparsers.add_parser('plot', help='Plot data distribution')
    plot_parser.add_argument('--filepath', type=str, required=True, help='Path to the input HDF5 file for plotting')
    plot_parser.add_argument('--true_keys', type=str, default='true_training,true_validation,true_test', help='Comma-separated list of keys for true data')
    plot_parser.add_argument('--fake_keys', type=str, default='fake_training,fake_validation,fake_test', help='Comma-separated list of keys for fake data')

    args = parser.parse_args()

    # Execute based on the chosen subcommand
    if args.command == 'umap':
        reduce_dimensions(args.filepath_true, args.filepath_fake, args.dim, args.neighbors, args.word_embedding, sample_size=args.sample_size)
    elif args.command == 'plot':
        plot_file(args.filepath, args.true_keys.split(','), args.fake_keys.split(','))
    else:
        parser.print_help()


#  python.exe .\umap_explore.py umap --filepath_true="dataset/ISOT/True_bert.h5" --filepath_fake="dataset/ISOT/Fake_bert.h5" --word_embedding='bert' --dim=2 --neighbors=700 --sample_size=1000
# python.exe .\umap_explore.py plot --filepath="dataset/ISOT/True_Fake_bert_umap_2dim_700_1000.h5"
if __name__ == '__main__':
    main()
