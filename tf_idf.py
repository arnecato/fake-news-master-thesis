import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import nltk
import umap
from nltk.corpus import words
import os
import h5py
import warnings
warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)
english_words = set(words.words())

umap_sample_size = 4000
dim = 2
neighbors = 4000
min_dist=0.0
metric='euclidean'
sample_size = 21417 # TODO: do for full dataset

files_to_check = [
    f'dataset/ISOT/fake_test_tfidf_cleaned_{sample_size}.hd5',
    f'dataset/ISOT/true_test_tfidf_cleaned_{sample_size}.hd5',
    f'dataset/ISOT/training_tfidf_cleaned_{sample_size}.hd5'
]

if not all(os.path.exists(file) for file in files_to_check):
    print('Cleaning files...')
    true_df = pd.read_csv('dataset/ISOT/True.csv')
    fake_df = pd.read_csv('dataset/ISOT/Fake.csv')
    true_df = true_df.sample(sample_size, random_state=42)
    fake_df = fake_df.sample(sample_size, random_state=42)
    true_training_df, true_test_df = train_test_split(true_df, test_size=0.3, random_state=42)
    fake_training_df, fake_test_df = train_test_split(fake_df, test_size=0.3, random_state=42)
    training_df = pd.concat([true_training_df.assign(label='true'), fake_training_df.assign(label='fake')]).reset_index(drop=True)
    #training_df = training_df.sample(sample_size, random_state=42)
    training_df = training_df[['label','title', 'text']]
    training_df['text'] = training_df.apply(lambda row: f"{row['title']}. {row['text']}", axis=1)
    training_df = training_df[['label', 'text']]
    training_df['text'] = training_df['text'].apply(lambda x: ' '.join(word for word in str(x).split() if word.lower() in english_words and word.lower() not in nltk.corpus.stopwords.words('english')))
    true_test_df['text'] = true_test_df['text'].apply(lambda x: ' '.join(word for word in str(x).split() if word.lower() in english_words and word.lower() not in nltk.corpus.stopwords.words('english')))
    fake_test_df['text'] = fake_test_df['text'].apply(lambda x: ' '.join(word for word in str(x).split() if word.lower() in english_words and word.lower() not in nltk.corpus.stopwords.words('english')))
    training_df.to_hdf(f'dataset/ISOT/training_tfidf_cleaned_{sample_size}.hd5', key='df', mode='w')
    true_test_df.to_hdf(f'dataset/ISOT/true_test_tfidf_cleaned_{sample_size}.hd5', key='df', mode='w')
    fake_test_df.to_hdf(f'dataset/ISOT/fake_test_tfidf_cleaned_{sample_size}.hd5', key='df', mode='w')
else:
    print('Cleaned files already exist!')
    training_df = pd.read_hdf(f'dataset/ISOT/training_tfidf_cleaned_{sample_size}.hd5', key='df')
    true_test_df = pd.read_hdf(f'dataset/ISOT/true_test_tfidf_cleaned_{sample_size}.hd5', key='df')
    fake_test_df = pd.read_hdf(f'dataset/ISOT/fake_test_tfidf_cleaned_{sample_size}.hd5', key='df')

files_to_check = [
    f'dataset/ISOT/true_training_tfidf_{sample_size}.pkl',
    f'dataset/ISOT/fake_training_tfidf_{sample_size}.pkl',
    f'dataset/ISOT/true_test_tfidf_{sample_size}.pkl',
    f'dataset/ISOT/fake_test_tfidf_{sample_size}.pkl'
]

if not all(os.path.exists(file) for file in files_to_check):
    print('Creating tf-idf vectors...')
    tfidf_vectorizer = TfidfVectorizer(analyzer='word',stop_words= 'english', token_pattern='(?u)\\b[a-zA-Z]{2,}\\b')

    training_documents = training_df['text'].tolist()
    # convert into matrix
    tfidf_wm_training = tfidf_vectorizer.fit_transform(training_documents)
    # Convert the tf-idf matrix to numpy arrays
    training_df['vector'] = [vec.toarray().flatten() for vec in tfidf_wm_training]
    #training_df = pd.read_hdf('dataset/ISOT/tmp.hdf5', key='df')
    #print('my vector', training_df.iloc[0]['vector'])
    #print('my vector', training_df.iloc[0]['vector'][100])
    #for i, vec in enumerate(training_df['vector'][0]):
    #    print(f'Index {i}: {vec}')  
    # convert into matrix using the tf-idf model from the training data
    true_test_documents = true_test_df['text'].tolist()
    fake_test_documents = fake_test_df['text'].tolist()
    tfidf_wm_true_test = tfidf_vectorizer.transform(true_test_documents)
    true_test_df['vector'] = [vec.toarray().flatten() for vec in tfidf_wm_true_test]
    tfidf_wm_fake_test = tfidf_vectorizer.transform(fake_test_documents)
    fake_test_df['vector'] = [vec.toarray().flatten() for vec in tfidf_wm_fake_test]
    #print(fake_test_df.iloc[0]['vector'], 'type:', type(fake_test_df.iloc[0]['vector']))
    #print('my vector at spot 0', fake_test_df.iloc[0]['vector'][40])

    # Separate the dataframes again
    true_training_df = training_df[training_df['label'] == 'true'].drop(columns=['label'])
    fake_training_df = training_df[training_df['label'] == 'fake'].drop(columns=['label'])
    print(true_training_df.head(), type(true_training_df.iloc[0]['vector']), len(true_training_df.iloc[0]['vector']))
    # Save the true training dataframe in chunks
    
    # Serialize the vectors into strings
    #true_training_df['vector'] = true_training_df['vector'].apply(lambda x: ','.join(map(str, x)))
    # Save the DataFrame
    true_training_df.to_pickle(f'dataset/ISOT/true_training_tfidf_{sample_size}.pkl')
    fake_training_df.to_pickle(f'dataset/ISOT/fake_training_tfidf_{sample_size}.pkl')
    true_test_df.to_pickle(f'dataset/ISOT/true_test_tfidf_{sample_size}.pkl')
    fake_test_df.to_pickle(f'dataset/ISOT/fake_test_tfidf_{sample_size}.pkl')
else:
    print('TF-IDF vectors already exist!')
    true_training_df = pd.read_pickle(f'dataset/ISOT/true_training_tfidf_{sample_size}.pkl')
    fake_training_df = pd.read_pickle(f'dataset/ISOT/fake_training_tfidf_{sample_size}.pkl')
    true_test_df = pd.read_pickle(f'dataset/ISOT/true_test_tfidf_{sample_size}.pkl')
    fake_test_df = pd.read_pickle(f'dataset/ISOT/fake_test_tfidf_{sample_size}.pkl')

true_fake_umap_fitting_df = pd.concat([true_training_df.sample(int(umap_sample_size/2), random_state=42), fake_training_df.sample(int(umap_sample_size/2), random_state=42)])
print('UMAP fitting size:', len(true_fake_umap_fitting_df))
dimension_reducer = umap.UMAP(n_components=dim, n_neighbors=neighbors, n_jobs=-1, min_dist=min_dist, metric=metric) 

dimension_reducer.fit(np.vstack(true_fake_umap_fitting_df['vector'].values))
# reduce dimensions of all training data
reduced_true_training_vectors = dimension_reducer.transform(np.vstack(true_training_df['vector'].values))
true_training_df['vector'] =  [np.array(vec) for vec in reduced_true_training_vectors]
reduced_true_test_vectors = dimension_reducer.transform(np.vstack(true_test_df['vector'].values))
true_test_df['vector'] =  [np.array(vec) for vec in reduced_true_test_vectors]
# reduce dimensions for all fake data
reduced_fake_training_vectors = dimension_reducer.transform(np.vstack(fake_training_df['vector'].values))
fake_training_df['vector'] = [np.array(vec) for vec in reduced_fake_training_vectors]
reduced_fake_test_vectors = dimension_reducer.transform(np.vstack(fake_test_df['vector'].values))
fake_test_df['vector'] = [np.array(vec) for vec in reduced_fake_test_vectors]

#fake_df = pd.read_hdf(filepath_fake, key='df')
#fake_df['vector'] = fake_df['vector'].apply(lambda x: x / np.linalg.norm(x))
# all fake data can be transformed at once
#reduced_vectors = dimension_reducer.transform(np.vstack(fake_df['vector'].values))
#fake_df['vector'] = [np.array(vec) for vec in reduced_vectors]

# save to file
#if sample_size == -1:
#    sample_size = 'all'
#if postfix != '':
#    postfix = '_' + postfix
filepath = f'dataset/ISOT/True_Fake_tf-idf_umap_{dim}dim_{neighbors}_{umap_sample_size}_{sample_size}.h5'
# add metadata
with h5py.File(filepath, 'a') as f:
    # Add metadata at the file level
    f.attrs['description'] = 'ISOT Dataset with True and Fake news'
    f.attrs['word_embedding'] = 'tf-idf'
    f.attrs['dim'] = dim
    f.attrs['neighbors'] = neighbors
    f.attrs['sample_size'] = sample_size
true_training_df.to_hdf(filepath, key='true_training', mode='w')
#true_validation_df.to_hdf(filepath, key='true_validation', mode='a')
true_test_df.to_hdf(filepath, key='true_test', mode='a')
fake_training_df.to_hdf(filepath, key='fake_training', mode='a')
fake_test_df.to_hdf(filepath, key='fake_test', mode='a')