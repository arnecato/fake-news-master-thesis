import numpy as np
import itertools
from detectors import Detector, DetectorSet
from util import visualize_2d, visualize_3d, precision, recall, euclidean_distance, circle_overlap_area, fast_cosine_distance_with_radius
import pandas as pd
import os
import time
import argparse

'''
NSGAII part based on Deb (2001)
'''


def compute_fitness(self, detector_set):
    ''' Fitness function to be used by the detector '''
    area = np.pi * self.radius**2
    overlap = 0
    if detector_set is not None:
        for detector in detector_set.detectors:
            if not np.array_equal(self.vector, detector.vector):          
                overlap += circle_overlap_area(self.vector, self.radius, detector.vector, detector.radius) #TODO: check for issue with negative radius - impact?
    self.f1 = area - overlap
    self.f2 = overlap

class NSGAII_Negative_Selection():

    def __init__(self, dim, pop_size, self_region_rate, true_df, detector_set, distance_type='euclidean'):
        self.dim = dim
        self.pop_size = pop_size
        #self.mutation_rate = mutation_rate
        self.true_df = true_df
        self.detector_set = detector_set
        self.distance_type = distance_type
        # find mean and stdev of feature values
        feature_values = [value for value in self.true_df['vector'].tolist()]
        #self.feature_mean = np.mean(true_df['vector'].tolist(), axis=0)
        #self.feature_stdev = np.std(true_df['vector'].tolist(), axis=0)
        self.feature_low = np.min(true_df['vector'].tolist(), axis=0)
        self.feature_max = np.max(true_df['vector'].tolist(), axis=0)
        self.range = self.feature_max - self.feature_low
        self.feature_max = self.feature_max + self.range * 0.1
        self.feature_low = self.feature_low - self.range * 0.1
        #self.feature_stdev = self.feature_stdev / 2 # TODO: find better way to limit the stdev. Stdev is used to bound the total space area (when initializing detectors)
        print('Feature values (low, max):', self.feature_low, self.feature_max)
        # find mean and stdev of distances between random detectors and self samples
        #self.euclidean_distance_mean, self.euclidean_distance_std, self.cosine_distance_mean, self.cosine_distance_std = self.find_distributions()
        #print('Euclidean distance (mean,stdev):', self.euclidean_distance_mean, self.euclidean_distance_std, 'Cosine (mean,stdev):', self.cosine_distance_mean, self.cosine_distance_std)
        self_distances = []
        for row_1st in self.true_df.itertuples(index=False, name=None):
            closest_distance = 999999.0
            for row_2nd in self.true_df.itertuples(index=False, name=None):
                distance = euclidean_distance(row_1st[1], row_2nd[1], 0, 0)
                if distance < closest_distance and distance != 0:
                    closest_distance = distance
            #print(f'distance found {closest_distance}', row_1st[1], row_2nd[1])
            if distance != 0: # avoid adding distance to itself
                self_distances.append(closest_distance)
        self.self_region = np.mean(self_distances) * self_region_rate
        print('Self region:', self.self_region)
        # total space
        #TODO: check this code
        self.total_space = np.prod(self.feature_max - self.feature_low)
        tmp_positive_space = len(true_df) * np.pi * self.self_region**2
        tmp_self_overlap = 0
        for i, row_1st in enumerate(self.true_df.itertuples(index=False, name=None)):
            for j, row_2nd in enumerate(self.true_df.itertuples(index=False, name=None)):
                if i != j:
                    tmp_self_overlap += circle_overlap_area(row_1st[1], self.self_region, row_2nd[1], self.self_region) / 2.0 # only count overlap for one of them
        self.total_positive_space = tmp_positive_space - tmp_self_overlap
        self.total_negative_space = self.total_space - self.total_positive_space   
        print('Total space:', self.total_space, 'Total positive space:', self.total_positive_space, 'Total negative space:', self.total_negative_space) 

        # M.O Initialization
        self.pareto_fronts_log = []
        self.pareto_fronts = []
        self.f1_max = -np.inf
        self.f1_min = np.inf
        self.f2_max = -np.inf
        self.f2_min = np.inf

    def initiate_population(self):
        # adds random parents and a batch of random "offspring"
        self.population =  [Detector.create_detector(self.feature_low, self.feature_max, self.dim, self.true_df, self.self_region, self.detector_set, self.distance_type, compute_fitness) for _ in range(self.pop_size * 2)]
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
        self.initiate_population()
        # sort based on fitness
        pareto_fronts = self.non_domination_sorting()
        best_f1 = 0
        best_f2 = 99999
        stagnant = 0
        #mutation_change = mutation_change_rate * (self.feature_mean + self.feature_stdev * 3)
        #mutation_max = np.maxself.feature_mean + self.feature_stdev * 3
        #print('Mutation change', mutation_change, 'Mutation max', mutation_max)
        generations = 0
        pareto_fronts = None
        while stagnant < stagnation_patience:

            self.pareto_fronts_log.append(pareto_fronts)
            mating_pool = self.create_mating_pool()
            pair_wise_parents = self.crowded_tournament_selection()
            offspring_pool = self.create_offspring_pool(pair_wise_parents)
   
            # mutate offspring
            for offspring in offspring_pool:
                #print('offspring', offspring, offspring.distance_type)
                # mutate (or move away from nearest)
                step = np.random.beta(a=1, b=4) # more likely to draw closer to 0
                distance_to_detector, nearest_detector = Detector.compute_closest_detector(self.detector_set, offspring.vector, self.distance_type)
                distance_to_self, nearest_self = Detector.compute_closest_self(self.true_df, self.self_region, offspring.vector, self.distance_type)
                if distance_to_self < distance_to_detector:
                    nearest_vector = nearest_self
                else:
                    nearest_vector = nearest_detector
                offspring.move_away_from_nearest(nearest_vector, step, self.feature_low, self.feature_max)
                distance_to_detector, nearest_detector = Detector.compute_closest_detector(self.detector_set, offspring.vector, self.distance_type)
                distance_to_self, nearest_self = Detector.compute_closest_self(self.true_df, self.self_region, offspring.vector, self.distance_type)
                offspring.radius = np.min([distance_to_detector, distance_to_self])
                offspring.compute_fitness(self.detector_set)
            
            # create random detectors to fill up to 2*pop_size
            # 50 parents, 25 offspring, 25 random
            refill = self.pop_size * 2 - len(mating_pool) - len(offspring_pool)
            random_pool = [Detector.create_detector(self.feature_low, self.feature_max, self.dim, self.true_df.sample(int(pop_check_ratio * len(self.true_df))), self.self_region, self.detector_set, self.distance_type, compute_fitness) for _ in range(refill)]
            for rnd_detector in random_pool:
                rnd_detector.compute_fitness(self.detector_set)

            # set population to be parents + offspring + random
            self.population = np.concatenate((mating_pool, offspring_pool, random_pool))
            pareto_fronts = self.non_domination_sorting()                

            # update stagnation (to decide on convergence)
            #TODO: implement check for f2 as well
            detector = self.pareto_fronts[0].individuals[0]
            if True: #generations % 10 == 0:
                print(detector.f1, np.max(detector.vector), np.min(detector.vector))
            if abs(detector.f1 - best_f1) > 0.01:
                best_f1 = detector.f1
                stagnant = 0
            else:
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
        return offspring_pool

class ParetoFront():
    ''' creates sorted list for each objective '''
    def __init__(self, outer_population, individuals, individual_indexes):
        self.pop = outer_population # reference to outer class population array
        self.individual_indexes = individual_indexes
        self.individuals = individuals
        #print(self.individuals)
        self.f1_sorted_list = sorted(self.individuals, key=lambda individual: individual.f1)
        self.f2_sorted_list = sorted(self.individuals, key=lambda individual: individual.f2, reverse=True)
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
    parser.add_argument('--self_region_rate', type=float, default=1.0, help='Rate to adjust the self region size')
    args = parser.parse_args()

    #dataset_file = f'dataset/ISOT/True_Fake_{args.word_embedding}_umap_{args.dim}dim_{args.neighbors}_{args.samples}.h5'
    true_training_df = pd.read_hdf(args.dataset, key='true_training')
    true_validation_df = pd.read_hdf(args.dataset, key='true_validation')
    true_test_df = pd.read_hdf(args.dataset, key='true_test')

    fake_training_df = pd.read_hdf(args.dataset, key='fake_training')
    fake_validation_df = pd.read_hdf(args.dataset, key='fake_validation')
    fake_test_df = pd.read_hdf(args.dataset, key='fake_test')

    if os.path.exists(args.detectorset):
        print(f"Detectors already exists. Expanding mature detector set > {args.detectorset}")
        dset = DetectorSet.load_from_file(args.detectorset, compute_fitness) 
    else:
        print(f"Detectors do not exist. Building mature detector set from scratch > {args.detectorset}")
        dset = DetectorSet([])
    
    #TODO: make population size hyperparameter (args.pop_size)
    nsga_nsa = NSGAII_Negative_Selection(args.dim, 10, args.self_region_rate, true_training_df, dset, 'euclidean')

    for i in range(args.amount):
        pareto_fronts = nsga_nsa.evolve_detector(2, pop_check_ratio=1) 
        best_f1 = 0
        new_detector = None
        for i, front in enumerate(pareto_fronts):
            print('Front:', i)
            for detector in front.individuals:
                print('Detector:', detector.vector, detector.radius, detector.f1, detector.f2)
                if i == 0: # pareto front 0 is the best front
                    if detector.f1 > best_f1:
                        best_f1 = detector.f1
                        new_detector = detector
        #new_detector = pareto_fronts[0].individuals[-1]
        print('Picking detector from pareto front', new_detector.vector, new_detector.radius, new_detector.f1, new_detector.f2)        
        dset.detectors.append(new_detector)
        print('Detectors:', len(dset.detectors))
        time0 = time.perf_counter()
        dset.save_to_file(args.detectorset)
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

    dset = DetectorSet.load_from_file(args.detectorset, compute_fitness)
    print('Detectors:', len(dset.detectors))
    #for detector in dset.detectors:
    #    detector.radius = detector.radius * 1
    time0 = time.perf_counter()
    true_detected, true_total = nsga_nsa.detect(pd.concat([true_validation_df, true_test_df]), dset, 99999)
    fake_detected, fake_total = nsga_nsa.detect(pd.concat([fake_validation_df, fake_test_df]), dset, 99999)
    #true_detected, true_total = nsga.detect(true_df, dset, 9999)
    #fake_detected, fake_total = nsga.detect(fake_df, dset, 9999)
    print(time.perf_counter() - time0)
    print('Precision:', precision(fake_detected, true_detected), 'Recall', recall(fake_detected, fake_total - fake_detected))
    print('True/Real detected:', true_detected, 'Total real/true:', true_total, 'Fake detected:', fake_detected, 'Total fake:', fake_total)

    # Visualize true, fake and detectors
    #detector_positions = np.array([detector.vector for detector in dset.detectors])
    #true_cluster = np.array(true_validation_df['vector'].tolist())
    #fake_cluster = np.array(fake_validation_df['vector'].tolist())
    if args.dim == 2:
        visualize_2d(pd.concat([true_validation_df, true_test_df]), pd.concat([fake_validation_df, fake_test_df]), dset, nsga_nsa.self_region)

    if args.dim == 3:
        visualize_3d(pd.concat([true_validation_df, true_test_df]), pd.concat([fake_validation_df, fake_test_df]), dset, nsga_nsa.self_region)    

if __name__ == "__main__":
    main()
