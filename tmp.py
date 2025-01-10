import pandas as pd
import h5py

filepath = f'dataset/ISOT/True_Fake_tf-idf_umap_{2}dim_{4000}_{4000}_{21417}.h5'
# add metadata
with h5py.File(filepath, 'a') as f:
    # Add metadata at the file level
    f.attrs['description'] = 'ISOT Dataset with True and Fake news'
    f.attrs['word_embedding'] = 'tf-idf'
    f.attrs['dim'] = 2
    f.attrs['neighbors'] = 4000
    f.attrs['sample_size'] = 21417
