import numpy as np
from ga_nsa import NegativeSelectionGeneticAlgorithm, DetectorSet, Detector
import pandas as pd
import time
from util import precision, recall, visualize_2d, visualize_3d
from sklearn.decomposition import PCA
import umap
from vectorfactory import SpacyVectorFactory
from bert import BERTVectorFactory
from sklearn.model_selection import train_test_split



def generate_cluster(center, num_points, spread):
    return np.random.normal(center, spread, size=(num_points, 2))

dim = 2
word_embedding = 'bert' # 'glove' or 'bert'
#vec_factory = BERTVectorFactory() # SpacyVectorFactory() # BERTVectorFactory() # SpacyVectorFactory() #BERTVectorFactory()
true_df = pd.read_hdf(f'dataset/ISOT/True_{word_embedding}_umap_{dim}dim.h5', key='df') 
# center vectors
umap_embeddings_true = np.vstack(true_df['vector'].values)
mean_true = umap_embeddings_true.mean(axis=0)
umap_embeddings_true_centered = umap_embeddings_true - mean_true
true_df['vector'] = [np.array(vec) for vec in umap_embeddings_true_centered] # TODO: move this code to dataset processing. UMAP, centering etc.
'''true_df = vec_factory.load_vectorized_dataframe('dataset/ISOT/True_glove.h5')
true_df = true_df.sample(1000, random_state=42)
dimension_reducer = umap.UMAP(n_components=dim) #PCA(n_components=dim)
umap_reducer = umap.UMAP(n_components=dim, n_jobs=-1)
vectors_df = pd.DataFrame(true_df['vector'].tolist())
reduced_vectors = dimension_reducer.fit_transform(vectors_df)

true_df['vector'] = [np.array(vec) for vec in reduced_vectors]
true_df.to_hdf(f'dataset/ISOT/True_glove_umap_{dim}dim.h5', key='df', mode='w')'''
print('Dim reduced to:', len(true_df.iloc[0]['vector']))
true_training_df, tmp_df = train_test_split(true_df, test_size=0.4, random_state=42)
true_validation_df, true_test_df = train_test_split(tmp_df, test_size=0.5, random_state=42)

fake_df = pd.read_hdf(f'dataset/ISOT/Fake_{word_embedding}_umap_{dim}dim.h5', key='df')
# center fake vectors based on true mean
umap_embeddings_fake = np.vstack(fake_df['vector'].values)
umap_embeddings_fake_centered = umap_embeddings_fake - mean_true
fake_df['vector'] = [np.array(vec) for vec in umap_embeddings_fake_centered]

'''fake_df = vec_factory.load_vectorized_dataframe('dataset/ISOT/Fake_glove.h5')
fake_df = fake_df.sample(1000, random_state=42)
vectors_df = pd.DataFrame(fake_df['vector'].tolist())
reduced_vectors = dimension_reducer.transform(vectors_df)
fake_df['vector'] = [np.array(vec) for vec in reduced_vectors]
fake_df.to_hdf(f'dataset/ISOT/Fake_glove_umap_{dim}dim.h5', key='df', mode='w')'''
fake_training_df, tmp_df = train_test_split(fake_df, test_size=0.4, random_state=42)
fake_validation_df, fake_test_df = train_test_split(tmp_df, test_size=0.5, random_state=42)

# 
dset = DetectorSet.load_from_file('detectors_test_2dim_umap_bert_some_overlap.json') # DetectorSet.load_from_file('detectors_test_2dim_umap_bert.json') # DetectorSet([]) # DetectorSet.load_from_file('detectors_test_3dim_cosine.json')  # DetectorSet([]) # DetectorSet.load_from_file('detectors_test_3dim.json')  # DetectorSet.load_from_file('detectors_test_4dim.json')  # DetectorSet([]) #DetectorSet.load_from_file('detectors_test.json') 
nsga = NegativeSelectionGeneticAlgorithm(dim, 15, 1, 0.02, true_training_df, dset, 'euclidean')
for i in range(0):
    detector = nsga.evolve_detector(2, pop_check_ratio=1, mutation_change_rate=0.0001) 
    if detector.radius > 0:
        dset.detectors.append(detector)
    print('Detectors:', len(dset.detectors))
    time0 = time.perf_counter()
    dset.save_to_file('detectors_test_2dim_umap_bert_some_overlap.json')
    print(time.perf_counter() - time0)
    print(len(dset.detectors))
    '''if len(dset.detectors) % 10 == 0:
        print('Detectors:', len(dset.detectors))
        time0 = time.perf_counter()
        true_detected, true_total = nsga.detect(true_df, dset, 9999)
        fake_detected, fake_total = nsga.detect(fake_df, dset, 9999)
        print(time.perf_counter() - time0)
        print('Precision:', precision(fake_detected, true_detected), 'Recall', recall(fake_detected, fake_total - fake_detected))
        print('True/Real detected:', true_detected, 'Total real/true:', true_total, 'Fake detected:', fake_detected, 'Total fake:', fake_total)
        if dim == 2:
            visualize_2d(true_df, fake_df, dset)
        elif dim == 3:
            visualize_3d(true_df, fake_df, dset)'''


dset = DetectorSet.load_from_file('detectors_test_2dim_umap_bert_some_overlap.json')
print('Detectors:', len(dset.detectors))
#for detector in dset.detectors:
#    detector.radius = detector.radius * 1
time0 = time.perf_counter()
true_detected, true_total = nsga.detect(pd.concat([true_validation_df, true_test_df]), dset, 9999)
fake_detected, fake_total = nsga.detect(pd.concat([fake_validation_df, fake_test_df]), dset, 9999)
print(time.perf_counter() - time0)
print('Precision:', precision(fake_detected, true_detected), 'Recall', recall(fake_detected, fake_total - fake_detected))
print('True/Real detected:', true_detected, 'Total real/true:', true_total, 'Fake detected:', fake_detected, 'Total fake:', fake_total)

# Visualize true, fake and detectors
#detector_positions = np.array([detector.vector for detector in dset.detectors])
#true_cluster = np.array(true_validation_df['vector'].tolist())
#fake_cluster = np.array(fake_validation_df['vector'].tolist())
if dim == 2:
    visualize_2d(true_df, fake_df, dset, nsga.self_region)

if dim == 3:
    visualize_3d(true_df, fake_df, dset, nsga.self_region)    