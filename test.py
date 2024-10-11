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

dim = 5
neighbors = 300
word_embedding = 'bert' # 'glove' or 'bert'
samples = 1000
dataset_file = f'dataset/ISOT/True_Fake_{word_embedding}_umap_{dim}dim_{neighbors}_{samples}.h5'
true_training_df = pd.read_hdf(dataset_file, key='true_training')
true_validation_df = pd.read_hdf(dataset_file, key='true_validation')
true_test_df = pd.read_hdf(dataset_file, key='true_test')

fake_training_df = pd.read_hdf(dataset_file, key='fake_training')
fake_validation_df = pd.read_hdf(dataset_file, key='fake_validation')
fake_test_df = pd.read_hdf(dataset_file, key='fake_test')

# DetectorSet([])     DetectorSet.load_from_file(f'detectors_test_{dim}dim_umap_{word_embedding}_{neighbors}_{samples}.json')
dset = DetectorSet.load_from_file(f'detectors_test_{dim}dim_umap_{word_embedding}_{neighbors}_{samples}.json') # DetectorSet.load_from_file(f'detectors_test_{dim}dim_umap_{word_embedding}_{neighbors}_{samples}.json') # DetectorSet.load_from_file(f'detectors_test_{dim}dim_umap_{word_embedding}_{neighbors}_{samples}.json')   #DetectorSet.load_from_file(f'detectors_test_{dim}dim_umap_{word_embedding}_{neighbors}_{samples}.json')# DetectorSet.load_from_file(f'detectors_test_{dim}dim_umap_{word_embedding}_{neighbors}_{min_dist}.json')#  DetectorSet.load_from_file(f'detectors_test_{dim}dim_umap_{word_embedding}_{neighbors}_{min_dist}.json') # DetectorSet.load_from_file(f'detectors_test_{dim}dim_umap_{word_embedding}_{neighbors}_{min_dist}.json') # DetectorSet.load_from_file('detectors_test_2dim_umap_bert.json') # DetectorSet([]) # DetectorSet.load_from_file('detectors_test_3dim_cosine.json')  # DetectorSet([]) # DetectorSet.load_from_file('detectors_test_3dim.json')  # DetectorSet.load_from_file('detectors_test_4dim.json')  # DetectorSet([]) #DetectorSet.load_from_file('detectors_test.json') 
nsga = NegativeSelectionGeneticAlgorithm(dim, 50, 1, 1, true_training_df, dset, 'euclidean')
for i in range(0):
    detector = nsga.evolve_detector(2, pop_check_ratio=1, mutation_change_rate=0.0001) 
    if detector.f1 > 0:
        dset.detectors.append(detector)
    print('Detectors:', len(dset.detectors))
    time0 = time.perf_counter()
    dset.save_to_file(f'detectors_test_{dim}dim_umap_{word_embedding}_{neighbors}_{samples}.json')
    '''if len(dset.detectors) % 1 == 0:
        print('Detectors:', len(dset.detectors))
        time0 = time.perf_counter()
        true_detected, true_total = nsga.detect(true_df, dset, 9999)
        fake_detected, fake_total = nsga.detect(fake_df, dset, 9999)
        print(time.perf_counter() - time0)
        print('Precision:', precision(fake_detected, true_detected), 'Recall', recall(fake_detected, fake_total - fake_detected))
        print('True/Real detected:', true_detected, 'Total real/true:', true_total, 'Fake detected:', fake_detected, 'Total fake:', fake_total)
        if dim == 2:
            visualize_2d(true_df, fake_df, dset, nsga.self_region)
        elif dim == 3:
            visualize_3d(true_df, fake_df, dset, nsga.self_region)'''

dset = DetectorSet.load_from_file(f'detectors_test_{dim}dim_umap_{word_embedding}_{neighbors}_{samples}.json')
print('Detectors:', len(dset.detectors))
#for detector in dset.detectors:
#    detector.radius = detector.radius * 1
time0 = time.perf_counter()
true_detected, true_total = nsga.detect(pd.concat([true_validation_df, true_test_df]), dset, 9999)
fake_detected, fake_total = nsga.detect(pd.concat([fake_validation_df, fake_test_df]), dset, 9999)
#true_detected, true_total = nsga.detect(true_df, dset, 9999)
#fake_detected, fake_total = nsga.detect(fake_df, dset, 9999)
print(time.perf_counter() - time0)
print('Precision:', precision(fake_detected, true_detected), 'Recall', recall(fake_detected, fake_total - fake_detected))
print('True/Real detected:', true_detected, 'Total real/true:', true_total, 'Fake detected:', fake_detected, 'Total fake:', fake_total)

# Visualize true, fake and detectors
#detector_positions = np.array([detector.vector for detector in dset.detectors])
#true_cluster = np.array(true_validation_df['vector'].tolist())
#fake_cluster = np.array(fake_validation_df['vector'].tolist())
if dim == 2:
    visualize_2d(pd.concat([true_validation_df, true_test_df]), pd.concat([fake_validation_df, fake_test_df]), dset, nsga.self_region)

if dim == 3:
    visualize_3d(pd.concat([true_validation_df, true_test_df]), pd.concat([fake_validation_df, fake_test_df]), dset, nsga.self_region)    