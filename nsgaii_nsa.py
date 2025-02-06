import numpy as np
import random
import itertools
from detectors import Detector, DetectorSet
from util import visualize_2d, visualize_3d, precision, recall, euclidean_distance, calculate_radius_overlap, fast_cosine_distance_with_radius, total_detector_hypersphere_volume, calculate_self_region, hypersphere_volume, hypersphere_overlap
import pandas as pd
import os
import time
import argparse
import json
import h5py
import math

'''
NSGAII part based on Deb et al. (2001)
'''

def compute_fitness(self, detector_set, nonself):
    ''' Fitness function to be used by the detector '''
    overlap_volume = 0
    dim = len(self.vector)
    volume = hypersphere_volume(self.radius, dim)
    closest_detector = None
    closest_distance = float('inf')
    if detector_set is not None:
        for detector in detector_set.detectors:
            if not np.array_equal(self.vector, detector.vector):          
                overlap_volume += hypersphere_overlap(self.radius, detector.radius, math.dist(self.vector, detector.vector), dim) #calculate_radius_overlap(self.vector, self.radius, detector.vector, detector.radius) #TODO: check for issue with negative radius - impact?
                distance = euclidean_distance(self.vector, detector.vector, 0, detector.radius)
                if distance < closest_distance:
                    closest_distance = distance
                    closest_detector = detector
    self.f1 = volume - overlap_volume 

    '''closest_nonself_distance = float('inf')
    #nonself_sample = random.sample(list(nonself), min(10, len(nonself)))
    for nonself_vector in nonself:
        distance = euclidean_distance(self.vector, nonself_vector, 0, 0)
        if distance < closest_nonself_distance:
            #print('Closest nonself distance:', distance)
            closest_nonself_distance = distance
    self.f2 = closest_nonself_distance'''

    if closest_detector is not None:
        self.f2 = closest_detector.radius
    else:
        self.f2 = float('inf')

class NSGAII_Negative_Selection():

    def __init__(self, dim, pop_size, self_region_rate, true_df, detector_set, distance_type, self_region, nonself):
        self.dim = dim
        self.pop_size = pop_size
        self.self_points = np.array(true_df['vector'].tolist())
        self.non_self_points = np.array(nonself['vector'].sample(100).tolist())
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
        print(self_region)
        if self_region == -1:
            for row_1st in self.self_points: #.itertuples(index=False, name=None):
                closest_distance = 999999.0
                for row_2nd in self.self_points: #.itertuples(index=False, name=None):
                    distance = euclidean_distance(row_1st, row_2nd, 0, 0)
                    if distance < closest_distance and distance != 0:
                        closest_distance = distance
                #print(f'distance found {closest_distance}', row_1st[1], row_2nd[1])
                if distance != 0: # avoid adding distance to itself
                    self_distances.append(closest_distance)
            self.self_region = np.mean(self_distances) * self_region_rate
        else:
            self.self_region = self_region
        print('Self region:', self.self_region)

        # M.O Initialization
        self.pareto_fronts_log = []
        self.pareto_fronts = []
        self.f1_max = -np.inf
        self.f1_min = np.inf
        self.f2_max = np.inf
        self.f2_min = -np.inf

    def initiate_population(self):
        # adds random parents and a batch of random "offspring"
        self.population =  [Detector.create_detector(self.feature_low, self.feature_max, self.dim, self.self_points, self.self_region, self.detector_set, self.distance_type, compute_fitness, self.range) for _ in range(self.pop_size * 2)]
        for detector in self.population:
            detector.compute_fitness(self.detector_set)
        self.store_min_and_max()

    def reset_distances(self):
        for ind in self.population:
            ind.distance = 0

    '''def evolve_generation(self):
        pareto_fronts = self.non_domination_sorting()
        self.pareto_fronts_log.append(pareto_fronts)
        mating_pool = self.create_mating_pool()
        pair_wise_parents = self.crowded_tournament_selection()
        offspring_pool = self.create_offspring_pool(pair_wise_parents)
        self.pop = np.concatenate((mating_pool, offspring_pool))
        self.reset_distances()'''

    def evolve_detector(self, stagnation_patience=10, pop_check_ratio=0.1):
        #self.voronoi_points = calculate_voronoi_points(self.detector_set)
        self.initiate_population()
        # sort based on objective fitness
        pareto_fronts = self.non_domination_sorting()
        #best_f1 = 0
        #best_f2 = 99999
        best_f1 = 0.00000001
        best_f2 = 0.00000001
        stagnant = 0
        #mutation_change = mutation_change_rate * (self.feature_mean + self.feature_stdev * 3)
        #mutation_max = np.maxself.feature_mean + self.feature_stdev * 3
        #print('Mutation change', mutation_change, 'Mutation max', mutation_max)
        generations = 0
        pareto_fronts = None
        while stagnant < stagnation_patience:

            self.pareto_fronts_log.append(pareto_fronts)
            mating_pool = self.create_mating_pool() 
            #for detector in mating_pool:
            #    print('Mating pool detector:', detector.vector, detector.radius, detector.f1, detector.f2)
            pair_wise_parents = self.crowded_tournament_selection() #TODO: why need to re-find the parents? Already have a mating pool
            offspring_pool = self.create_offspring_pool(pair_wise_parents)
   
            # mutate offspring
            '''for offspring in offspring_pool:
                #print('offspring', offspring, offspring.distance_type)
                # mutate (or move away from nearest)
                step = np.random.beta(a=1, b=4) # more likely to draw closer to 0
                distance_to_detector, nearest_detector = Detector.compute_closest_detector(self.detector_set, offspring.vector, self.distance_type)
                distance_to_self, nearest_self, closest_self = Detector.compute_closest_self(self.self_points, self.self_region, offspring.vector, self.distance_type, self.range)
                if distance_to_self < distance_to_detector:
                    nearest_vector = nearest_self.copy()
                else:
                    nearest_vector = nearest_detector.vector.copy()
                offspring.move_away_from_nearest(nearest_vector, step, self.feature_low, self.feature_max)
                distance_to_detector, nearest_detector = Detector.compute_closest_detector(self.detector_set, offspring.vector, self.distance_type)
                distance_to_self, nearest_self, closest_self = Detector.compute_closest_self(self.self_points, self.self_region, offspring.vector, self.distance_type, self.range)
                offspring.radius = np.min([distance_to_detector, distance_to_self])
                offspring.compute_fitness(self.detector_set)'''
            # mutate
            for offspring in offspring_pool:
                # half of the time move away from closest
                if random.choice([True, False]):
                    distance_to_detector, nearest_detector = Detector.compute_closest_detector(self.detector_set, offspring.vector, self.distance_type)
                    distance_to_self, nearest_self, closest_selves = Detector.compute_closest_self(self.self_points, self.self_region, offspring.vector, self.distance_type, self.range)
                    if distance_to_self < distance_to_detector:
                        other_vector = nearest_self.copy()
                    else:
                        other_vector = nearest_detector.vector.copy()
                    
                    step = np.random.beta(a=1, b=4) # more likely to draw closer to 0
                    offspring.move_away_from_nearest(other_vector, step, self.feature_low, self.feature_max)
                # half of the time mutate normally
                else:
                    offspring.mutate(self.feature_low, self.feature_max)
                
                distance_to_detector, nearest_detector = Detector.compute_closest_detector(self.detector_set, offspring.vector, self.distance_type)
                distance_to_self, nearest_self, closest_selves = Detector.compute_closest_self(self.self_points, self.self_region, offspring.vector, self.distance_type, self.range)
                offspring.radius = np.min([distance_to_detector, distance_to_self])
                offspring.compute_fitness(self.detector_set)
            # create random detectors to fill up to 2*pop_size
            # 50 parents, 25 offspring, 25 random
            #TODO: remove random individuals, these should be created by offsprings instead
            #refill = self.pop_size * 2 - len(mating_pool) - len(offspring_pool)
            #random_pool = [Detector.create_detector(self.feature_low, self.feature_max, self.dim, self.self_points, self.self_region, self.detector_set, self.distance_type, compute_fitness, self.range) for _ in range(refill)]
            #for rnd_detector in random_pool:
            #    rnd_detector.compute_fitness(self.detector_set)

            # set population to be parents + offspring + random
            #print('Population size:', len(mating_pool), len(offspring_pool), len(random_pool))
            self.population = np.concatenate((mating_pool, offspring_pool))
            pareto_fronts = self.non_domination_sorting()                

            # update stagnation (to decide on convergence)
            #TODO: implement check for f2 as well
            #TODO: Important! need to do proper M.O convergence here (now it is just checking f1 convergence based on first detector in optimal pareto front)
            best_detector = max(self.pareto_fronts[0].individuals, key=lambda detector: detector.f1)
            # update stagnation (to decide on convergence)
            #best_detector = self.population[0]
            #print('Best detector:', best_detector.f1, best_detector.vector)
            best_detector.compute_fitness(self.detector_set)
            print(best_f1, best_detector.f1, np.max(best_detector.vector), np.min(best_detector.vector))
            
            if best_f1 != 0 and abs((best_detector.f1 - best_f1) / best_f1) > 0.001:
                best_f1 = best_detector.f1
                stagnant = 0
                #print('Stagnant reset')
            elif best_f2 != 0 and abs((best_detector.f2 - best_f2) / best_f2) > 0.001:
                best_f2 = best_detector.f2
                stagnant = 0
                #print('Stagnant reset')
            else:
                #print('Stagnant', stagnant)
                stagnant += 1
            
            generations += 1

        # TODO: add support for pop_check_ratio < 1
        if pop_check_ratio < 1:
            raise Exception('pop_check_ratio needs to be 1 - support of < 1 is not implemented')
        
        '''if pop_check_ratio < 1:
            old_r = self.population[0].radius
            distance_to_detector, nearest_detector = Detector.compute_closest_detector(self.detector_set, offspring.vector, self.distance_type)
            distance_to_self, nearest_self = Detector.compute_closest_self(self.true_df, self.self_region, offspring.vector, self.distance_type)
            self.population[0].radius = np.min([distance_to_detector, distance_to_self])
            print(self.population[0].vector)
            self.population[0].compute_fitness(self.detector_set)
            print('Changed radius:', old_r, self.population[0].radius)'''

        self.reset_distances()

        return pareto_fronts
    
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

    def store_min_and_max(self):
        ''' Stores max and min f_m values in population '''

        for ind in self.population:
            if ind.f1 > self.f1_max:
                self.f1_max = ind.f1
            if ind.f1 < self.f1_min:
                self.f1_min = ind.f1
            if ind.f2 > self.f2_max:
                self.f2_max = ind.f2
            if ind.f2 < self.f2_min:
                self.f2_min = ind.f2

    def non_domination_sorting(self):
        ''' Sorts the population into non-domination ranks. Rank 1 means non-dominated, rank 2 means domination by rank 1 individuals '''
        
        # [[0, 1], [0, 2], [1, 2]]
        # iterate through population (P) and new offspring (Q) and compare all against all, put them in pareto fronts
        indexes = np.arange(0, self.pop_size*2)
        comparison_list = list(itertools.combinations(indexes, 2))
        comparison_result = [[] for _ in range(self.pop_size*2)] # preparing list of individuals dominating individual 
        
        for comp_indexes in comparison_list:
            #print(len(self.population), comp_indexes)
            if self.population[comp_indexes[0]].dominated_by(self.population[comp_indexes[1]]):
                comparison_result[comp_indexes[0]].append(comp_indexes[1])
            elif self.population[comp_indexes[1]].dominated_by(self.population[comp_indexes[0]]):
                comparison_result[comp_indexes[1]].append(comp_indexes[0])

        # find pareto fronts
        self.pareto_fronts = []     
        c = self.pop_size*2
        while c > 0:
            optimal_pareto_individual_indexes = []
            for i, comp_result in enumerate(comparison_result):
                if comp_result != None and len(comp_result) == 0:
                    optimal_pareto_individual_indexes.append(i)
                    comparison_result[i] = None
            c -= len(optimal_pareto_individual_indexes)
            self.pareto_fronts.append(ParetoFront(self, [self.population[i] for i in optimal_pareto_individual_indexes], optimal_pareto_individual_indexes))
            comparison_result = [[item for item in sublist if item not in self.pareto_fronts[-1].individual_indexes] if sublist is not None else None for sublist in comparison_result]
        return self.pareto_fronts

    def create_mating_pool(self):
        mating_pool = np.empty(self.pop_size, dtype=Detector)
        f_counter = 0 # front counter
        c = 0 # individual counter
        for f in range(len(self.pareto_fronts)):
            f_counter = f
            if len(self.pareto_fronts[f].individuals_sorted_by_distance) + c < self.pop_size:
                for ind in self.pareto_fronts[f].individuals_sorted_by_distance:
                    mating_pool[c] = ind
                    c += 1
            else:
                break
        
        last_front_individuals = []
        for ind in self.pareto_fronts[f_counter].individuals_sorted_by_distance:
            mating_pool[c] = ind
            last_front_individuals.append(ind)
            c += 1
            if c >= self.pop_size:
                break 

        self.pareto_fronts = self.pareto_fronts[0:f_counter]
        self.pareto_fronts.append(ParetoFront(self, last_front_individuals, None))

        return mating_pool        
    
    def crowded_tournament_selection(self):
        ''' creates a pairwise list of parents to be recombined '''
        pareto_fronts_participating = np.random.choice(np.arange(0, len(self.pareto_fronts)), self.pop_size*2, replace=True)
        pairwise_parents = np.empty(self.pop_size, dtype=Detector)
        c = 0 
        for i in range(0,len(pareto_fronts_participating)-1, 2):
            front_a_index = pareto_fronts_participating[i]
            front_b_index = pareto_fronts_participating[i+1]
            if front_a_index < front_b_index:
                t = self.pareto_fronts[front_a_index].individuals
                pairwise_parents[c] = self.pareto_fronts[front_a_index].individuals[np.random.randint(len(self.pareto_fronts[front_a_index].individuals))]
            elif front_a_index > front_b_index:
                pairwise_parents[c] = self.pareto_fronts[front_b_index].individuals[np.random.randint(len(self.pareto_fronts[front_b_index].individuals))]
            else:
                # same pareto front, find individual with largest crowding distance, winner is the lowest from crowding sorted individuals
                winner_index = np.min(np.random.choice(np.arange(0, len(self.pareto_fronts[pareto_fronts_participating[i]].individuals_sorted_by_distance)), 2, replace=True)) # EFFICIENCY? WHAT IF ONLY ONE INDIVIDUAL IN A FRONT? 
                pairwise_parents[c] = self.pareto_fronts[pareto_fronts_participating[i]].individuals_sorted_by_distance[winner_index]
            c += 1 
        
        return pairwise_parents

    def create_offspring_pool(self, pairwise_parents):
        offspring_pool = [] # np.empty(len(pairwise_parents), dtype=Detector)
        for i in range(0, len(pairwise_parents)-1, 2):
            offspring_pool.extend(pairwise_parents[i].recombine(pairwise_parents[i+1], compute_fitness))
            offspring_pool.extend(pairwise_parents[i].recombine(pairwise_parents[i+1], compute_fitness))
        return offspring_pool

class ParetoFront():
    ''' creates sorted list for each objective '''
    def __init__(self, outer_population, individuals, individual_indexes):
        self.pop = outer_population # reference to outer class population array
        self.individual_indexes = individual_indexes
        self.individuals = individuals
        #print(self.individuals)
        self.f1_sorted_list = sorted(self.individuals, key=lambda individual: individual.f1)
        self.f2_sorted_list = sorted(self.individuals, key=lambda individual: individual.f2) # TODO: made it a maximize goal instead. Need to double check this later!
        self.calculate_crowding_distance()

    def calculate_crowding_distance(self):
        ''' calculates the crowding distance for individuals in each front '''
            # Ensuring the max and min values are set
        #print(self.pop)
        #for individual in self.pop.population:
        #    print(f"f1: {individual.f1}, f2: {individual.f2}")
        self.pop.f2_min = min(individual.f2 for individual in self.pop.population)
        self.pop.f2_max = max(individual.f2 for individual in self.pop.population)
        self.pop.f1_min = min(individual.f1 for individual in self.pop.population)
        self.pop.f1_max = max(individual.f1 for individual in self.pop.population)
        
        # set boundary solutions
        self.f1_sorted_list[0].distance = np.inf
        self.f2_sorted_list[0].distance = np.inf
        self.f1_sorted_list[-1].distance = np.inf
        self.f2_sorted_list[-1].distance = np.inf

        # iterate through front and set crowding distance
        for i in range(1, len(self.f1_sorted_list)-1):
            #print('f1 min', self.pop.f1_min, 'f1 max', self.pop.f1_max, 'f2 min', self.pop.f2_min, 'f2 max', self.pop.f2_max)
            self.f1_sorted_list[i].distance = self.f1_sorted_list[i].distance + (self.f1_sorted_list[i+1].f1 - self.f1_sorted_list[i-1].f1) / (self.pop.f1_max - self.pop.f1_min)
        if self.pop.f2_max != self.pop.f2_min:
            for i in range(1, len(self.f2_sorted_list)-1):
                self.f2_sorted_list[i].distance = self.f2_sorted_list[i].distance + (self.f2_sorted_list[i+1].f2 - self.f2_sorted_list[i-1].f2) / (self.pop.f2_max - self.pop.f2_min)

        # sort based on crowding distance
        self.individuals_sorted_by_distance = sorted(self.individuals, key=lambda individual: -individual.distance)            

# python nsgaii.py --dim=2 --dataset=dataset\ISOT\True_Fake_bert_umap_2dim_600_1000.h5 --detectorset=model\detector\detectors_bert_2dim_600_1000.h5 --amount=1
def main():
    parser = argparse.ArgumentParser(description='Multi-Objective Negative Selection Algorithm for Fake News Detection')
    parser.add_argument('--dim', type=int, default=5, help='Dimensionality of the feature vectors')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset file')
    parser.add_argument('--detectorset', type=str, required=True, help='Path to the detectorset file')
    parser.add_argument('--amount', type=int, required=True, help='Amount of detectors to evolve')
    parser.add_argument('--sample', type=int, default=-1, help='Number of samples to use from the dataset')
    parser.add_argument('--self_region', type=float, default=-1, help='Self region size')
    parser.add_argument('--self_region_rate', type=float, default=1.0, help='Rate to adjust the self region size')
    parser.add_argument('--convergence_every', type=int, default=10, help='Check for convergence every x iterations')
    parser.add_argument('--coverage', type=float, default=0.005, help='Increase in coverage threshold for deciding convergence')
    parser.add_argument('--auto', type=int, default=0, help='Whether to run in auto mode (0 for False, 1 for True)')
    parser.add_argument('--experiment', type=int, default=-1, help='Experiment number')
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
        true_training_df = true_training_df.sample(args.sample, random_state=500)
    #true_validation_df = pd.read_hdf(args.dataset, key='true_validation')
    true_test_df = pd.read_hdf(args.dataset, key='true_test')

    fake_training_df = pd.read_hdf(args.dataset, key='fake_training')
    #fake_validation_df = pd.read_hdf(args.dataset, key='fake_validation')
    fake_test_df = pd.read_hdf(args.dataset, key='fake_test')
    args.detectorset = args.detectorset.replace('.json', '') 
    args.detectorset = f'{args.detectorset}_{args.experiment}.json'
    if os.path.exists(args.detectorset):
        print(f"Detectors already exists. Expanding mature detector set > {args.detectorset}")
        dset = DetectorSet.load_from_file(args.detectorset, compute_fitness) 
    else:
        print(f"Detectors do not exist. Building mature detector set from scratch > {args.detectorset}")
        dset = DetectorSet([])
    
    #TODO: make population size hyperparameter (args.pop_size)
    nsga_nsa = NSGAII_Negative_Selection(args.dim, 10, args.self_region_rate, true_training_df, dset, 'euclidean', args.self_region, fake_training_df)
    last_detector_negative_coverage = total_detector_hypersphere_volume(dset)
    coverage_over_time = []
    time0 = time.perf_counter()
    for i in range(args.amount):
        pareto_fronts = nsga_nsa.evolve_detector(3, pop_check_ratio=1) 
        best_f1 = 0
        new_detector = None
        for i, front in enumerate(pareto_fronts):
            #print('Front:', i)
            for detector in front.individuals:
                if i == 0: # pareto front 0 is the best front
                    #print('Front 0 - Detector:', detector.vector, detector.radius, detector.f1, detector.f2)
                    if detector.f1 > best_f1:
                        best_f1 = detector.f1
                        new_detector = detector
        #new_detector = pareto_fronts[0].individuals[-1]
        if new_detector is not None:
            print('Picking detector from pareto front', new_detector.vector, new_detector.radius, new_detector.f1, new_detector.f2, len(dset.detectors) + 1)      
            if new_detector.f1 > 0:
                dset.detectors.append(new_detector)
            else:
                print('Detector not added, f1 < 0! -----------------------------------------------------------------------------------------! ')
        else:
            print('No detector found ------------------------------------------------------------------------------------------------------- !')
        # check for convergence
        #print('Negative space coverage:', total_detector_hypersphere_volume(dset))
        if len(dset.detectors) % args.convergence_every == 0 and len(dset.detectors) > 0 and i > 0:
            negative_space_coverage = total_detector_hypersphere_volume(dset)
            print('Negative space coverage:', negative_space_coverage)            
            coverage_over_time.append(negative_space_coverage)
            if last_detector_negative_coverage > 0:
                coverage_pct = (negative_space_coverage - last_detector_negative_coverage) / last_detector_negative_coverage
                if coverage_pct > args.coverage:
                    print('Checking for convergence', negative_space_coverage, last_detector_negative_coverage, coverage_pct)
                    print(coverage_over_time)
                    dset.save_to_file(args.detectorset)
                else:
                    print('Converged', negative_space_coverage, last_detector_negative_coverage, coverage_pct)
                    print(coverage_over_time)
                    break
            last_detector_negative_coverage = negative_space_coverage
    time_to_build = time.perf_counter() - time0
    print('Total time to build model:', time_to_build)
    if os.path.exists(args.detectorset):
        print(f"Loading existing detectors from {args.detectorset}")
        dset = DetectorSet.load_from_file(args.detectorset, compute_fitness)
    else:
        print(f"Detectors file not found. Relying on existing dset")
    print('Detectors:', len(dset.detectors))
    #for detector in dset.detectors:
    #    detector.radius = detector.radius * 1
    time0 = time.perf_counter()
    #real_test_set_df = pd.concat([true_validation_df, true_test_df])
    #fake_test_set_df = pd.concat([fake_validation_df, fake_test_df])
    #print('TESTING ON TRAINING DATASET!! ******')
    true_detected, true_total = nsga_nsa.detect(true_test_df, dset, 9999) # TODO: switch back to test!
    fake_detected, fake_total = nsga_nsa.detect(fake_test_df, dset, 9999)# TODO: switch back to test!
    #true_detected, true_total = nsga.detect(true_df, dset, 9999)
    #fake_detected, fake_total = nsga.detect(fake_df, dset, 9999)
    time_to_infer = time.perf_counter() - time0
    print('Precision:', precision(fake_detected, true_detected), 'Recall', recall(fake_detected, fake_total - fake_detected))
    print('True/Real detected:', true_detected, 'Total real/true:', true_total, 'Fake detected:', fake_detected, 'Total fake:', fake_total)
    
    # generate test results
    true_detected_list = []
    fake_detected_list = []
    precision_list = []
    recall_list = []
    negative_space_coverage_list = []
    for i in range(100, len(dset.detectors), 100):
        tmp_dset = DetectorSet(dset.detectors[:i])
        negative_space_coverage_list.append(total_detector_hypersphere_volume(tmp_dset))
        true_detected, true_total = nsga_nsa.detect(true_test_df, dset, i)
        fake_detected, fake_total = nsga_nsa.detect(fake_test_df, dset, i)
        true_detected_list.append(true_detected)
        fake_detected_list.append(fake_detected)
        precision_list.append(precision(fake_detected, true_detected))
        recall_list.append(recall(fake_detected, fake_total - fake_detected))
        print('Precision:', precision(fake_detected, true_detected), 'Recall', recall(fake_detected, fake_total - fake_detected))

    results = {
        "precision": precision(fake_detected, true_detected),
        "recall": recall(fake_detected, fake_total - fake_detected),
        "true_detected": true_detected,
        "true_total": true_total,
        "fake_detected": fake_detected,
        "fake_total": fake_total,
        "negative_space_coverage": total_detector_hypersphere_volume(dset),
        "time_to_build": time_to_build,
        "detectors_count": len(dset.detectors),
        "precision_list": precision_list,
        "recall_list": recall_list,
        "true_detected_list": true_detected_list,
        "fake_detected_list": fake_detected_list,
        "negative_space_coverage_list": negative_space_coverage_list,
        "time_to_infer": time_to_infer,
        "self_region": nsga_nsa.self_region
    }

    experiment_filepath = args.detectorset.replace('.json', '') + f'_experiment_results_{args.experiment}.json'
    with open(experiment_filepath, 'w') as f:
        json.dump(results, f, indent=4)  

    true_plot_df = true_training_df # true_training_df #true_test_df # real_test_set_df
    fake_plot_df = fake_training_df # fake_training_df 
    
    # only plot if not in auto mode
    if args.auto == 0:
        if args.dim == 2:
            visualize_2d(true_plot_df, fake_plot_df, dset, nsga_nsa.self_region)

        if args.dim == 3:
            visualize_3d(true_plot_df, fake_plot_df, dset, nsga_nsa.self_region)   

if __name__ == "__main__":
    main()


# python .\ga_nsa.py --dim=2 --dataset=dataset/ISOT/True_Fake_bert_umap_2dim_4000_4000_10708.h5 --detectorset=model\detector\detectors_bert_2dim_4000_4000_10708.json --amount=1 --convergence_every=100 --self_region=0.021507087160302772