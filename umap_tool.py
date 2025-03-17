import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import umap
from util import visualize_3d, calculate_self_region

from detectors import DetectorSet
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# Ensure LaTeX rendering for professional output
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 12
})

import time
import h5py
import os
import warnings
import pickle
warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)

def euclidean_distance(a, b, a_radius, b_radius):
    return np.linalg.norm(a - b) - a_radius - b_radius

def save_self_region():
    dims = [2,3,4]
    embeddings = ['bert-base-cased', 'fasttext'] #'distilbert-base-cased', 
    # True_Fake_bert-base-cased_umap_4dim_4000_4000_21417.h5
    for dim in dims:
        for embedding in embeddings:
            filepath = f'dataset/ISOT/True_Fake_{embedding}_umap_{dim}dim_4000_4000_21417.h5'
            print(filepath)
            true_training_df = pd.read_hdf(filepath, key='true_training')
            self_points = np.array(true_training_df['vector'].tolist())
            self_region = calculate_self_region(self_points)
            with h5py.File(filepath, 'a') as f:
                f.attrs['self_region'] = self_region
            print('Self region:', self_region, 'File:', filepath)

def reduce_dimensions(filepath_true, filepath_fake, dim, neighbors, word_embedding, sample_size=1000, umap_sample_size=1000, min_dist=0.0, postfix='', metric='euclidean', spread=1.0):
    time0 = time.perf_counter()

    
    # check if file exists  
    # save to filepath
    if sample_size == -1:
        sample_size = 'all'
    if postfix != '':
        postfix = '_' + postfix
    filepath = f'dataset/ISOT/True_Fake_{word_embedding}_umap_{dim}dim_{neighbors}_{umap_sample_size}_{sample_size}{postfix}.h5'
    if not os.path.exists(filepath):
        true_df = pd.read_hdf(filepath_true, key='df') 
        fake_df = pd.read_hdf(filepath_fake, key='df')
        # create unit vectors
        #true_df['vector'] = true_df['vector'].apply(lambda x: x / np.linalg.norm(x))
        #fake_df['vector'] = fake_df['vector'].apply(lambda x: x / np.linalg.norm(x))
        #print('First vector in true_df:', true_df['vector'].iloc[0])
        true_df['label'] = 'true'
        fake_df['label'] = 'false'
        
        if sample_size == -1:
            sample_size = len(true_df)
        print('Sample size:', sample_size)
        true_df = true_df.sample(sample_size, random_state=42)
        fake_df = fake_df.sample(sample_size, random_state=42)
        if umap_sample_size == -1:
            umap_sample_size = sample_size

        #true_df['vector'] = true_df['vector'].apply(lambda x: x / np.linalg.norm(x))
        # TODO: REMOVE HARDCODING
        #true_df = true_df.sample(8000, random_state=42)
        print('Metrics:', metric, 'Min dist:', min_dist, 'Neighbors:', neighbors, 'Spread:', spread)
        true_training_df, true_test_df = train_test_split(true_df, test_size=0.4, random_state=42)
        true_validation_df, true_test_df = train_test_split(true_test_df, test_size=0.5, random_state=42)
        fake_training_df, fake_test_df = train_test_split(fake_df, test_size=0.4, random_state=42)
        fake_validation_df, fake_test_df = train_test_split(fake_test_df, test_size=0.5, random_state=42)
        print('True training size:', len(true_training_df), 'True validation size:', len(true_validation_df), 'True test size:', len(true_test_df))
        print('Fake training size:', len(fake_training_df), 'Fake validation size:', len(fake_validation_df), 'Fake test size:', len(fake_test_df))
        # prepare dimension reducer
        # CHECK OUT THIS CODE - USE FAKE NEWS OR NOT!?
        umap_training_df = pd.concat([true_training_df.sample(int(umap_sample_size/2), random_state=42), fake_training_df.sample(int(umap_sample_size/2), random_state=42)]) # TODO: Consider using FAKE data too!
        print('UMAP fitting size:', len(umap_training_df))
        dimension_reducer = umap.ParametricUMAP(n_components=dim, n_neighbors=neighbors, n_jobs=-1, min_dist=min_dist, metric=metric, spread=spread) 
        #dim_reducer_training_df = true_training_df.sample(sample_size, random_state=42) # TODO: REMOVE HARDCODING
        dimension_reducer.fit(np.vstack(umap_training_df['vector'].values))

         # Save the dimension reducer model
        model_name = f'{word_embedding}_{dim}dim'

        model_dir = 'model'
        model_path = os.path.join(model_dir, model_name)
        dimension_reducer.save(model_path)
        print(f'Dimension reducer model saved to {model_path}')

        # reduce dimensions of all training data
        '''reduced_true_training_vectors = dimension_reducer.transform(np.vstack(true_training_df['vector'].values))
        true_training_df['vector'] =  [np.array(vec) for vec in reduced_true_training_vectors]
        
        reduced_fake_training_vectors = dimension_reducer.transform(np.vstack(fake_training_df['vector'].values))
        fake_training_df['vector'] = [np.array(vec) for vec in reduced_fake_training_vectors]

        reduced_true_test_vectors = dimension_reducer.transform(np.vstack(true_test_df['vector'].values))
        true_test_df['vector'] = [np.array(vec) for vec in reduced_true_test_vectors]

        reduced_fake_test_vectors = dimension_reducer.transform(np.vstack(fake_test_df['vector'].values))
        fake_test_df['vector'] = [np.array(vec) for vec in reduced_fake_test_vectors]'''
        # concat version
        true_fake_training_df = pd.concat([true_training_df, fake_training_df])
        reduced_true_fake_training_vectors = dimension_reducer.transform(np.vstack(true_fake_training_df['vector'].values))

        true_fake_training_df['vector'] =  [np.array(vec) for vec in reduced_true_fake_training_vectors]
        
        true_training_df['vector'] = true_fake_training_df.loc[true_fake_training_df['label'] == 'true', 'vector']
        fake_training_df['vector'] = true_fake_training_df.loc[true_fake_training_df['label'] == 'false', 'vector']

        true_fake_test_df = pd.concat([true_test_df, fake_test_df])
        reduced_true_fake_test_vectors = dimension_reducer.transform(np.vstack(true_fake_test_df['vector'].values))
        true_fake_test_df['vector'] =  [np.array(vec) for vec in reduced_true_fake_test_vectors]
        true_test_df['vector'] = true_fake_test_df.loc[true_fake_test_df['label'] == 'true', 'vector']
        fake_test_df['vector'] = true_fake_test_df.loc[true_fake_test_df['label'] == 'false', 'vector']
        
        true_fake_validation_df = pd.concat([true_validation_df, fake_validation_df])
        reduced_true_fake_validation_vectors = dimension_reducer.transform(np.vstack(true_fake_validation_df['vector'].values))
        true_fake_validation_df['vector'] = [np.array(vec) for vec in reduced_true_fake_validation_vectors]
        true_validation_df['vector'] = true_fake_validation_df.loc[true_fake_validation_df['label'] == 'true', 'vector']
        fake_validation_df['vector'] = true_fake_validation_df.loc[true_fake_validation_df['label'] == 'false', 'vector']
        # reduce dimensions for all fake data
        #reduced_fake_training_vectors = dimension_reducer.transform(np.vstack(fake_training_df['vector'].values))
        #fake_training_df['vector'] = [np.array(vec) for vec in reduced_fake_training_vectors]
        #reduced_fake_test_vectors = dimension_reducer.transform(np.vstack(fake_test_df['vector'].values))
        #fake_test_df['vector'] = [np.array(vec) for vec in reduced_fake_test_vectors]

        #fake_df = pd.read_hdf(filepath_fake, key='df')
        #fake_df['vector'] = fake_df['vector'].apply(lambda x: x / np.linalg.norm(x))
        # all fake data can be transformed at once
        #reduced_vectors = dimension_reducer.transform(np.vstack(fake_df['vector'].values))
        #fake_df['vector'] = [np.array(vec) for vec in reduced_vectors]
        
        true_training_df.to_hdf(filepath, key='true_training', mode='w')
        true_validation_df.to_hdf(filepath, key='true_validation', mode='a')
        true_test_df.to_hdf(filepath, key='true_test', mode='a')
        fake_training_df.to_hdf(filepath, key='fake_training', mode='a')
        fake_validation_df.to_hdf(filepath, key='fake_validation', mode='a')
        fake_test_df.to_hdf(filepath, key='fake_test', mode='a')
        
        # find self region radius
        self_points = np.array(true_training_df['vector'].tolist())
        self_region = calculate_self_region(self_points)

        # add metadata
        with h5py.File(filepath, 'a') as f:
            # Add metadata at the file level
            f.attrs['description'] = 'ISOT Dataset with True and Fake news'
            f.attrs['word_embedding'] = word_embedding
            f.attrs['dim'] = dim
            f.attrs['neighbors'] = neighbors
            f.attrs['sample_size'] = sample_size
            f.attrs['self_region'] = self_region

        print('Dim reduced to:', len(true_training_df.iloc[0]['vector']), 'File', filepath, 'Processing time:', time.perf_counter()-time0)
    else:
        print('File exists:', filepath, 'Skipping processing')
    
def plot_file(filepath, true_keys, fake_keys, sample_size, embedding_model):
    true_df = pd.read_hdf(filepath, key=true_keys[0])

    for key in true_keys[1:]:
        true_df = pd.concat([true_df, pd.read_hdf(filepath, key=key)])
    fake_df = pd.read_hdf(filepath, key=fake_keys[0])
    for key in fake_keys[1:]:
        fake_df = pd.concat([fake_df, pd.read_hdf(filepath, key=key)])
    if sample_size != -1:
        true_df = true_df.sample(sample_size, random_state=42)
        fake_df = fake_df.sample(sample_size, random_state=42)

    # read metadata from hdf5 file
    with h5py.File(filepath, 'r') as f:
        word_embedding = f.attrs['word_embedding']
        dim = f.attrs['dim']
        neighbors = f.attrs['neighbors']
        sample_size = f.attrs['sample_size']
    plot(true_df, fake_df, word_embedding, neighbors, sample_size, true_keys, fake_keys, embedding_model)

def plot(true_df, fake_df, word_embedding, neighbors, sample_size, true_keys, fake_keys, embedding_model):
    true_vectors = np.vstack(true_df['vector'].values)
    fake_vectors = np.vstack(fake_df['vector'].values)
    
    if fake_vectors.shape[1] == 2:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=300)  # High-res for print
        
        ax.scatter(true_vectors[:, 0], true_vectors[:, 1], s=20, color='blue', alpha=0.25, label="Real News")
        ax.scatter(fake_vectors[:, 0], fake_vectors[:, 1], s=20, color='red', alpha=0.25, label="Fake News")
        
        ax.set_title(rf'UMAP {word_embedding} 2D - {len(true_vectors)+len(fake_vectors)} samples', fontsize=14, fontweight='bold')
        #ax.set_xlabel(r'UMAP 1', fontsize=12)
        #ax.set_ylabel(r'UMAP 2', fontsize=12)
        
        # Ticks formatting
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        
        ax.grid(True, linestyle='--', linewidth=0.5)
        ax.legend(loc='best', fontsize=10)
        
        plt.savefig("report/{embedding_model}_2D_plot.pdf", bbox_inches='tight', format='pdf')  # Save as vector format
        plt.show()
    
    elif fake_vectors.shape[1] == 3:
        dset = DetectorSet([])
        visualize_3d(true_df, fake_df, dset, 0.01, embedding_model)


def main():
    parser = argparse.ArgumentParser(description='UMAP Dimensionality Reduction and Other Options')
    subparsers = parser.add_subparsers(dest='command', help='Sub-command to run')
   
    umap_parser = subparsers.add_parser('umap', help='Perform UMAP dimensionality reduction')
    umap_parser.add_argument('--filepath_true', type=str, required=True, help='Path to the input HDF5 file for true news')
    umap_parser.add_argument('--filepath_fake', type=str, required=True, help='Path to the input HDF5 file for fake news')
    umap_parser.add_argument('--dim', type=int, required=True, help='Number of dimensions for UMAP')
    umap_parser.add_argument('--word_embedding', type=str, required=True, choices=['distilbert-base-cased', 'roberta-base', 'bert-base-cased', 'fasttext', 'fasttext-supervised', 'roberta-large'], help='Type of word embedding')
    umap_parser.add_argument('--neighbors', type=int, default=15, help='Number of neighbors for UMAP')
    umap_parser.add_argument('--sample_size', type=int, default=-1, help='Sample size for the dataset')
    umap_parser.add_argument('--min_dist', type=float, default=0.0, help='Minimum distance for UMAP')
    umap_parser.add_argument('--postfix', type=str, default='', help='Postfix for the output file name')
    umap_parser.add_argument('--metric', type=str, default='euclidean', help='Metric for UMAP')
    umap_parser.add_argument('--umap_sample_size', type=int, default=1000, help='Sample size for UMAP fitting')
    umap_parser.add_argument('--spread', type=float, default=1.0, help='Spread for UMAP')
    plot_parser = subparsers.add_parser('plot', help='Plot data distribution')
    plot_parser.add_argument('--filepath', type=str, required=True, help='Path to the input HDF5 file for plotting')
    plot_parser.add_argument('--true_keys', type=str, default='true_training,true_validation,true_test', help='Comma-separated list of keys for true data')
    plot_parser.add_argument('--fake_keys', type=str, default='fake_training,fake_validation,fake_test', help='Comma-separated list of keys for fake data')
    plot_parser.add_argument('--sample_size', type=int, default=-1, help='Sample size for the dataset')
    plot_parser.add_argument('--model', type=str, default="", help='NLP model used. To be used for filename.')

    args = parser.parse_args()

    # Execute based on the chosen subcommand
    if args.command == 'umap':
        reduce_dimensions(args.filepath_true, args.filepath_fake, args.dim, args.neighbors, args.word_embedding, sample_size=args.sample_size, umap_sample_size=args.umap_sample_size, min_dist=args.min_dist, postfix=args.postfix, metric=args.metric, spread=args.spread)
    elif args.command == 'plot':
        plot_file(args.filepath, args.true_keys.split(','), args.fake_keys.split(','), args.sample_size, args.model)
    else:
        parser.print_help()

#  python.exe .\umap_tool.py umap --filepath_true="dataset/ISOT/True_bert.h5" --filepath_fake="dataset/ISOT/Fake_bert.h5" --word_embedding='bert' --dim=2 --neighbors=700 --sample_size=1000
# python.exe .\umap_tool.py plot --filepath="dataset/ISOT/True_Fake_bert_umap_2dim_700_1000.h5"
if __name__ == '__main__':
    main()
    #save_self_region()


