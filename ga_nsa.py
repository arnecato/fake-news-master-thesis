import pandas as pd
import numpy as np
import random
import math
import time
import argparse
import os
from typing import List
from detectors import Detector, DetectorSet
from util import euclidean_distance, fast_cosine_distance_with_radius, precision, recall, f1_score, fast_cosine_distance, visualize_3d, visualize_2d, calculate_radius_overlap, get_shared_feature_vectors, get_nearby_self, hypersphere_overlap, hypersphere_volume, total_detector_hypersphere_volume, calculate_self_region
import json
import h5py
from sklearn.preprocessing import MinMaxScaler
# 2D self region ---
# bert long 3600_6000 self region = 0.04305774469300755
# bert 1800_3000 self region = 0.05695028924327248
# bert 3600_6000 self region = 0.04318423289042272
# bert 6000_12000 self region = 0.034759770545002774
# bert 4800_8000 self region = Self region: 0.02911155687966354
# bert COSINE 3600_6000 self region = 0.04245865356215778
# bert 256 3600_6000 self region = 0.043809546306835416
# bert 256 1800_3000 self region = 0.05809979095059515
# word2vec 3600 6000 self region = 0.04100741142238599
# word2vec COSINE 3600 6000 self region = 0.041199497979408105
# glove 6000 12000 self region = 0.03344242578467898
# glove 3600_6000 self region = 0.043154863199135265
# 3D self region ---
# bert 3600_6000 self region = 0.10658193345223824
# 4D self region ---
# bert 3600_6000 self region = 0.16171675445897507

def compute_fitness(self, detector_set):
    overlap_volume = 0
    dim = len(self.vector)
    volume = hypersphere_volume(self.radius, dim)
    if detector_set is not None:
        for detector in detector_set.detectors:
            if not np.array_equal(self.vector, detector.vector):    
                #self_vector, detector_vector, shared_features = get_shared_feature_vectors(self.vector, self.feature_index, detector.vector, detector.feature_index)
                self_vector = self.vector.copy()
                detector_vector = detector.vector.copy()
                if self_vector is not None and detector_vector is not None:
                    overlap_volume += hypersphere_overlap(self.radius, detector.radius, math.dist(self_vector, detector_vector), dim) #calculate_radius_overlap(self_vector, self.radius, detector_vector, detector.radius) #TODO: replace this!?
    #print('radius - overlap', self.radius, overlap, (self.radius - overlap))
    self.f1 = volume - overlap_volume #self.radius - overlap # TODO: REMOVE THIS /2

class NegativeSelectionGeneticAlgorithm():
    def __init__(self, dim, pop_size, mutation_rate, self_region, self_region_rate, true_df, detector_set, distance_type):
        self.dim = dim
        self.pop_size = pop_size
        #self.mutation_rate = mutation_rate          
        self.self_points = np.array(true_df['vector'].tolist())
        #get_nearby_self(self.self_points, np.array([21,8]), 1)
        self.detector_set = detector_set
        self.distance_type = distance_type
        #self.feature_selection = feature_selection
        self.feature_low = np.min(self.self_points, axis=0)
        self.feature_max = np.max(self.self_points, axis=0)
        self.range = self.feature_max - self.feature_low
        print('Feature range:', self.range)
        self.feature_max = self.feature_max + self.range * 0.25
        self.feature_low = self.feature_low - self.range * 0.25
        #self.voronoi_points = calculate_voronoi_points(self.detector_set)
        #self.feature_stdev = self.feature_stdev / 2 # TODO: find better way to limit the stdev. Stdev is used to bound the total space area (when initializing detectors)
        print('Feature values (low, max):', self.feature_low, self.feature_max)
        # find mean and stdev of distances between random detectors and self samples
        #self.euclidean_distance_mean, self.euclidean_distance_std, self.cosine_distance_mean, self.cosine_distance_std = self.find_distributions()
        #print('Euclidean distance (mean,stdev):', self.euclidean_distance_mean, self.euclidean_distance_std, 'Cosine (mean,stdev):', self.cosine_distance_mean, self.cosine_distance_std)
        self_distances = []
        if self_region == -1:
            for self_point_1 in self.self_points: #.itertuples(index=False, name=None):
                closest_distance = 999999.0
                for self_point_2 in self.self_points: #.itertuples(index=False, name=None):
                    distance = euclidean_distance(self_point_1, self_point_2, 0, 0)
                    if distance < closest_distance and distance != 0:
                        closest_distance = distance
                #print(f'distance found {closest_distance}', row_1st[1], row_2nd[1])
                if distance != 0: # avoid adding distance to itself
                    self_distances.append(closest_distance)
            self.self_region = np.mean(self_distances) * self_region_rate
        else:
            self.self_region = self_region
        print('Self region:', self.self_region)
        
        
        # find total space, positive space and negative space
        '''self.total_space = np.prod(self.feature_max - self.feature_low)

        tmp_self_overlap = 0
        for i, self_point_1 in enumerate(self.self_points): #self.true_df.itertuples(index=False, name=None)):
            for j, self_point_2 in enumerate(self.self_points): #enumerate(self.true_df.itertuples(index=False, name=None)):
                if i != j:
                    tmp_self_overlap += hypersphere_overlap(self.self_region, self.self_region, math.dist(self_point_1, self_point_2), self.dim) / 2.0 #alculate_overlap(row_1st[1], self.self_region, row_2nd[1], self.self_region) / 2.0 # only count overlap for one of them
        self_volume = hypersphere_volume(self.self_region, self.dim)
        tmp_positive_space = len(self.self_points) * self_volume
        self.total_positive_space = tmp_positive_space - tmp_self_overlap
        self.total_negative_space = self.total_space - self.total_positive_space   
        print('Total space:', self.total_space, self.feature_max, self.feature_low, 'Total positive space:', self.total_positive_space, 'Total negative space:', self.total_negative_space) '''
        
    def initiate_population(self):
        self.population = [Detector.create_detector(self.feature_low, self.feature_max, self.dim, self.self_points, self.self_region, self.detector_set, self.distance_type, compute_fitness, self.range) for _ in range(self.pop_size)] 

    def find_distributions(self):
        random_detectors = [Detector.create_detector(self.feature_low, self.feature_max, self.dim, self.self_points, 0, None, self.distance_type, compute_fitness, self.range) for _ in range(10)]
        cosine_distances = []
        euclidean_distances = []
        for vector in random_detectors:
            for row in self.self_points.itertuples(index=False, name=None):
                #distances.append(np.linalg.norm(vector.vector - row[1])) #self.vector_factory.document_vector(row[0]))
                cosine_distances.append(fast_cosine_distance(vector.vector, row[1]))
                euclidean_distances.append(euclidean_distance(vector.vector, row[1], 0, 0))
        return np.mean(euclidean_distances), np.std(euclidean_distances), np.mean(cosine_distances), np.std(cosine_distances)
            
    #def compute_population_fitness(self):
    #    for detector in self.population:
    #        detector.compute_fitness(self.detector_set)

    def tournament_selection(self, tournament_size=2):
        # Randomly select tournament_size individuals and choose the best
        contenders = random.sample(self.population, tournament_size)
        return max(contenders, key=lambda individual: individual.f1)

    def recombine_tournament(self):
        offspring_list = []
        
        # Perform tournament selection to select poulation size
        for _ in range(self.pop_size):
            # Select two individuals via tournament selection
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            #print(parent1.vector, parent2.vector)
            # Recombine the two parents to create offspring
            offsprings = parent1.recombine(parent2, compute_fitness)
            
            # Add offsprings to the offspring list
            offspring_list.extend(offsprings)

            # Stop once we have offspring equal to half the population
            #if len(offspring_list) >= int(self.pop_size / 2):
            #    break

        return offspring_list

    def evolve_detector(self, stagnation_patience=10, pop_check_ratio=0.1):
        #self.voronoi_points = calculate_voronoi_points(self.detector_set)
        self.initiate_population()
        best = 0.000000001
        stagnant = 0
        #mutation_change = mutation_change_rate * (self.feature_mean + self.feature_stdev * 3)
        #mutation_max = np.maxself.feature_mean + self.feature_stdev * 3
        #print('Mutation change', mutation_change, 'Mutation max', mutation_max)
        generations = 0
        while stagnant < stagnation_patience:
            # sort based on fitness
            self.population = sorted(self.population, key=lambda detector: detector.f1, reverse=True)
            
            # recombine the best
            offsprings: List[Detector] = self.recombine_tournament()
            # mutate
            for offspring in offsprings:
    
                # half of the time move away from closest
                if random.choice([True, False]):
                    distance_to_detector, nearest_detector = Detector.compute_closest_detector(self.detector_set, offspring.vector, self.distance_type)
                    median_distance_to_self, nearest_self, closest_selves = Detector.compute_closest_self(self.self_points, self.self_region, offspring.vector, self.distance_type, self.range)
                    if median_distance_to_self < distance_to_detector:
                        other_vector = nearest_self.copy()
                    else:
                        other_vector = nearest_detector.vector.copy()
                    
                    step = np.random.beta(a=1, b=4) # more likely to draw closer to 0
                    offspring.move_away_from_nearest(other_vector, step, self.feature_low, self.feature_max)
                # half of the time mutate normally
                else:
                    offspring.mutate(self.feature_low, self.feature_max)
                
                distance_to_detector, nearest_detector = Detector.compute_closest_detector(self.detector_set, offspring.vector, self.distance_type)
                median_distance_to_self, nearest_self, closest_selves = Detector.compute_closest_self(self.self_points, self.self_region, offspring.vector, self.distance_type, self.range)
                offspring.radius = np.min([distance_to_detector, median_distance_to_self])
                offspring.compute_fitness(self.detector_set)

            self.population.extend(offsprings)     
            # add random detectors
            '''rnd_detectors = [Detector.create_detector(self.feature_low, self.feature_max, self.dim, self.self_points, self.self_region, self.detector_set, self.distance_type, compute_fitness, self.range) for _ in range(int(0.5 * self.pop_size))]
            for rnd_detector in rnd_detectors:
                #distance_to_detector, nearest_detector = Detector.compute_closest_detector(self.detector_set, offspring.vector, self.distance_type, offspring.feature_index)
                #distance_to_self, nearest_self = Detector.compute_closest_self(self.true_df, self.self_region, rnd_detector.vector, self.distance_type, rnd_detector.feature_index)
                #offspring.radius = np.min([distance_to_detector, distance_to_self])
                rnd_detector.compute_fitness(self.detector_set)
                #print('Random detector:', rnd_detector.radius, rnd_detector.f1, distance_to_self)
            self.population.extend(rnd_detectors)'''
            
            #self.compute_population_fitness()
            # sort full population
            #print(self.population[0].vector, self.population[0].radius, self.population[0].f1)
            self.population = sorted(self.population, key=lambda detector: detector.f1, reverse=True)           
            #print('Population', len(self.population))
            #for detector in self.population:
            #    print(detector.vector, detector.f1)
            # select only the best to survive to the next generation
            self.population = self.population[:self.pop_size]                   

            # update stagnation (to decide on convergence)
            detector = self.population[0]
            print(best, detector.f1, np.max(detector.vector), np.min(detector.vector))
            if best != 0 and abs((detector.f1 - best) / best) > 0.001:
                best = detector.f1
                stagnant = 0
                #print('Stagnant reset')
            else:
                #print('Stagnant', stagnant)
                stagnant += 1
            generations += 1

        # TODO: below needs to be added if pop_check_ratio < 1
        if pop_check_ratio < 1:
            old_r = self.population[0].radius
            #distance_to_detector, nearest_detector = Detector.compute_closest_detector(self.detector_set, self.population[0].vector, self.distance_type, self.population[0].feature_index)
            median_distance_to_self, nearest_self, closest_selves = Detector.compute_closest_self(self.self_points, self.self_region, self.population[0].vector, self.distance_type, self.range)
            self.population[0].radius = median_distance_to_self #np.min([distance_to_detector, distance_to_self])
            print(self.population[0].vector)
            self.population[0].compute_fitness(self.detector_set)
            print('Changed radius:', old_r, self.population[0].radius)
        print('Returning best detector:', self.population[0].f1)
        return self.population[0]          

    def detect(self, df, detector_set, max_detectors):
        detected = 0
        total = 0
        for row in df.itertuples(index=False, name=None):
            detectors_used = 0
            for detector in detector_set.detectors:
                #distance = np.linalg.norm(detector.vector - row[1]) - self.self_region #self.vector_factory.document_vector(row[0]))    
                if detector.distance_type == 'cosine':
                    distance = fast_cosine_distance_with_radius(detector.vector, row[1], detector.radius, 0)
                elif detector.distance_type == 'euclidean':
                    distance = euclidean_distance(detector.vector, row[1], detector.radius, 0)
                #print('distance', distance, detector.radius)
                if distance <= 0:
                    #print(distance, detector.vector)
                    detected += 1
                    break
                detectors_used += 1
                if detectors_used >= max_detectors:
                    break
            total += 1          
        return detected, total


# python .\ga_nsa.py --dim=2 --dataset=dataset\ISOT\True_Fake_bert_umap_2dim_3600_6000.h5 --detectorset=model\detector\detectors_bert_2dim_3600_6000.json --amount=1 --self_region=0.037725538859821446
def main():
    parser = argparse.ArgumentParser(description='Negative Selection Genetic Algorithm for Fake News Detection')
    parser.add_argument('--dim', type=int, default=5, help='Dimensionality of the feature vectors')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset file')
    parser.add_argument('--detectorset', type=str, required=True, help='Path to the detectorset file')
    parser.add_argument('--amount', type=int, required=True, help='Amount of detectors to evolve')
    #parser.add_argument('--feature_selection', type=int, default=0, help='Whether to use feature selection (0 for False, 1 or more indicates the number of features to select for a new detector)')
    parser.add_argument('--self_region', type=float, default=-1, help='Self region size')
    parser.add_argument('--self_region_rate', type=float, default=1.0, help='Rate to adjust the self region size')
    parser.add_argument('--sample', type=int, default=-1, help='Number of samples to use from the dataset')
    parser.add_argument('--convergence_every', type=int, default=10, help='Check for convergence every x iterations')
    parser.add_argument('--coverage', type=float, default=0.005, help='Increase in coverage threshold for deciding convergence')
    parser.add_argument('--auto', type=int, default=0, help='Whether to run in auto mode (0 for False, 1 for True)')
    parser.add_argument('--experiment', type=int, default=-1, help='Experiment number')
    parser.add_argument('--model', type=str, help='Model name')
    args = parser.parse_args()

    #dataset_file = f'dataset/ISOT/True_Fake_{args.word_embedding}_umap_{args.dim}dim_{args.neighbors}_{args.samples}.h5'
    true_training_df = pd.read_hdf(args.dataset, key='true_training')

    with h5py.File(args.dataset, 'r') as f:
        if 'self_region' in f.attrs:
            args.self_region = f.attrs['self_region']
            print(f'Self region found in dataset {args.self_region}')
        else:
            print('No self region found in dataset')
            if args.self_region == -1:
                print('Self region not provided. Calculating self region from dataset')
                args.self_region = calculate_self_region(np.array(true_training_df['vector'].tolist()))
                print(f'Calculated self region: {args.self_region}')                

    if args.sample > 0:
        true_training_df = true_training_df.sample(args.sample)
    true_validation_df = pd.read_hdf(args.dataset, key='true_validation')
    true_test_df = pd.read_hdf(args.dataset, key='true_test')

    fake_training_df = pd.read_hdf(args.dataset, key='fake_training')
    fake_validation_df = pd.read_hdf(args.dataset, key='fake_validation')
    fake_test_df = pd.read_hdf(args.dataset, key='fake_test')

    # normalize vectors
    '''scaler = MinMaxScaler(feature_range=(-1, 1))
    true_training_df['vector'] = [np.array(vec) for vec in scaler.fit_transform(np.vstack(true_training_df['vector'].values))]
    fake_training_df['vector'] = [np.array(vec) for vec in scaler.transform(np.vstack(fake_training_df['vector'].values))]
    true_test_df['vector'] = [np.array(vec) for vec in scaler.transform(np.vstack(true_test_df['vector'].values))]
    fake_test_df['vector'] = [np.array(vec) for vec in scaler.transform(np.vstack(fake_test_df['vector'].values))]'''

    args.detectorset = args.detectorset.replace('.json', '') 
    args.detectorset = f'{args.detectorset}_{args.experiment}.json'
    experiment_filepath = args.detectorset.replace('.json', '') + f'_experiment_results_{args.experiment}.json'

    if os.path.exists(args.detectorset) and args.auto == 0:
        print(f"Detectors already exists. Expanding mature detector set > {args.detectorset}")
        dset = DetectorSet.load_from_file(args.detectorset, compute_fitness) 
    else:
        dset = DetectorSet([])
        if args.auto == 0:
            print(f"Detectors do not exist. Building mature detector set from scratch > {args.detectorset}")
        else:
            print(f"Auto experiment, building mature detector set from scratch > {args.detectorset}")

    if not(args.auto == 1 and os.path.exists(experiment_filepath)):
        nsga = NegativeSelectionGeneticAlgorithm(args.dim, 20, 1, args.self_region, args.self_region_rate, true_training_df, dset, 'euclidean')
        if args.auto == 1:
            stagnation = 0
            best_validation_f1 = 0
            best_dset = DetectorSet([])
            validation_true_detected_list = []
            validation_fake_detected_list = []
            validation_precision_list = []
            validation_recall_list = []
            validation_negative_space_coverage_list = []
            convergence_checked_already = []
            time0 = time.perf_counter()
            while len(dset.detectors) < args.amount:
                detector = nsga.evolve_detector(3, pop_check_ratio=1) 
                if detector.f1 > 0:
                    dset.detectors.append(detector)        
                print('Detectors:', len(dset.detectors))

                # check for convergence
                if len(dset.detectors) % args.convergence_every == 0:
                    dset.save_to_file(args.detectorset)  

                    if len(dset.detectors) not in convergence_checked_already:
                        # validation check
                        print('Checking f1 on validation and convergence')
                        true_validation_detected, true_validation_total = nsga.detect(true_validation_df, dset, len(dset.detectors))
                        fake_validation_detected, fake_validation_total = nsga.detect(fake_validation_df, dset, len(dset.detectors))
                        f1_validation = f1_score(fake_validation_detected, true_validation_detected, fake_validation_total - fake_validation_detected)
                        validation_negative_space_coverage_list.append(total_detector_hypersphere_volume(dset))
                        validation_true_detected_list.append(true_validation_detected)
                        validation_fake_detected_list.append(fake_validation_detected)
                        validation_precision_list.append(precision(fake_validation_detected, true_validation_detected))
                        validation_recall_list.append(recall(fake_validation_detected, fake_validation_total - fake_validation_detected))
                        if f1_validation > best_validation_f1:
                            best_validation_f1 = f1_validation
                            best_dset.detectors = dset.detectors.copy()
                            stagnation = 0
                            print('Validation Precision:', precision(fake_validation_detected, true_validation_detected), 'Recall', recall(fake_validation_detected, fake_validation_total - fake_validation_detected), 'F1:', f1_validation)    
                        else:
                            stagnation += 1
                            print('Stagnation', stagnation)
                            print('Validation Precision:', precision(fake_validation_detected, true_validation_detected), 'Recall', recall(fake_validation_detected, fake_validation_total - fake_validation_detected), 'F1:', f1_validation)    
                            if stagnation >= 5:
                                print('Early stopping', best_validation_f1, len(best_dset.detectors))
                                break
                        # make sure we do not check convergence again for this exact amount of detectors
                        convergence_checked_already.append(len(dset.detectors))

            time_to_build = time.perf_counter() - time0
            print('Total time to build model:', time_to_build)
            if best_dset is not None and len(best_dset.detectors) > 1:
                best_dset.save_to_file(args.detectorset)
            dset = DetectorSet.load_from_file(args.detectorset, compute_fitness)
            print('Detectors:', len(dset.detectors))
            
            time0 = time.perf_counter()
            true_detected, true_total = nsga.detect(true_test_df, best_dset, len(best_dset.detectors))
            fake_detected, fake_total = nsga.detect(fake_test_df, best_dset, len(best_dset.detectors))
            test_time_to_infer = time.perf_counter() - time0
            print('Test Precision:', precision(fake_detected, true_detected), 'Recall', recall(fake_detected, fake_total - fake_detected))

            results = {
                "test_precision": precision(fake_detected, true_detected),
                "test_recall": recall(fake_detected, fake_total - fake_detected),
                "test_f1": f1_score(fake_detected, true_detected, fake_total - fake_detected),
                "test_true_detected": true_detected,
                "test_true_total": true_total,
                "test_fake_detected": fake_detected,
                "test_fake_total": fake_total,
                "test_negative_space_coverage": total_detector_hypersphere_volume(best_dset),
                "test_time_to_build": time_to_build,
                "test_detectors_count": len(best_dset.detectors),
                "test_time_to_infer": test_time_to_infer,
                "validation_precision_list": validation_precision_list,
                "validation_recall_list": validation_recall_list,
                "validation_true_detected_list": validation_true_detected_list,
                "validation_fake_detected_list": validation_fake_detected_list,
                "validation_negative_space_coverage_list": validation_negative_space_coverage_list,
                "self_region": nsga.self_region,
                "stagnation": stagnation
            }

            with open(experiment_filepath, 'w') as f:
                json.dump(results, f, indent=4)
        else:
            true_plot_df = true_training_df #true_test_df # real_test_set_df
            fake_plot_df = fake_training_df
        
            if args.dim == 2:
                visualize_2d(true_plot_df, fake_plot_df, dset, nsga.self_region, 'nsa-ga', args.model)

            if args.dim == 3:
                visualize_3d(true_plot_df, fake_plot_df, dset, nsga.self_region)      
    else:
        print('Auto mode enabled and experiment results already exists. Skipping experiment.')
                
if __name__ == "__main__":
    main()

# python .\ga_nsa.py --dim=2 --dataset=dataset\ISOT\True_Fake_bert_umap_2dim_5600_8000.h5 --detectorset=model\detector\detectors_bert_2dim_5600_8000.json --amount=2100 --convergence_every=100 --self_region=0.0313902143297709

# Total time to build model: 618.99655479996
'''Detectors: 1000
33.391967199975625
Precision: 0.86306334699023 Recall 0.8523187052598817
True/Real detected: 869 Total real/true: 6426 Fake detected: 5477 Total fake: 6426'''