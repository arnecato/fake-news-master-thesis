import pandas as pd
import numpy as np
import random
from vectorfactory import SpacyVectorFactory
#from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import time
import json
from bert import BERTVectorFactory
from sklearn.decomposition import PCA
from util import euclidean_distance, fast_cosine_distance_with_radius, precision, recall, fast_cosine_distance, visualize_3d, visualize_2d, circle_overlap_area

class Detector():
    def __init__(self, vector, radius, distance_type, detector_set):
        self.vector = vector
        self.radius = radius
        self.fitness = radius
        self.distance_type = distance_type
        self.compute_fitness(detector_set)

    @classmethod
    def create_detector(cls, feature_low, feature_high, dim, self_df, self_region, detector_set, distance_type):
        best_vector = np.random.uniform(low=feature_low, high=feature_high)
        exceeding_max = best_vector > feature_high
        exceeding_max_negative = best_vector < feature_low
        
        if np.any(exceeding_max) or np.any(exceeding_max_negative):
            print('exceeding max C',best_vector) 
        best_distance, nearest_detector = Detector.compute_closest_detector(detector_set, best_vector, distance_type)   
        #print('Creating new detector', mean, stdev, best_vector)
        if detector_set is not None and len(detector_set.detectors) > 0:
            for _ in range(10):
                vector = np.random.uniform(low=feature_low, high=feature_high)
                distance_to_detector, nearest_detector = Detector.compute_closest_detector(detector_set, vector, distance_type)
                if distance_to_detector > best_distance:
                    best_distance = distance_to_detector
                    best_vector = vector   
                    #print('new vector', best_distance, vector)
        detector = Detector(best_vector, 0, distance_type, detector_set)
        distance_to_detector, nearest_detector = Detector.compute_closest_detector(detector_set, best_vector, distance_type)
        distance_to_self, nearest_self = Detector.compute_closest_self(self_df, self_region, best_vector, distance_type)
        
        detector.radius = np.min([distance_to_detector, distance_to_self]) 
        #print('created new', detector.radius, detector.vector)
        detector.compute_fitness(detector_set)    

        exceeding_max = best_vector > feature_high
        exceeding_max_negative = best_vector < feature_low
        
        if np.any(exceeding_max) or np.any(exceeding_max_negative):
            print('exceeding max CREATE DETECTOR',best_vector)        
        return detector
    
    @classmethod
    #def compute_maximum_radius_detector_and_self(cls, self_df, self_region_radius, detector_set, vector, distance_type):
    #    #print('returning:', np.min([Detector.compute_maximum_radius(self_df, self_region_radius, vector, distance_type), Detector.compute_maximum_radius_to_detector_set(detector_set, vector, distance_type)]))
    #    return np.min([Detector.compute_maximum_radius(self_df, self_region_radius, vector, distance_type), Detector.compute_maximum_radius_to_detector_set(detector_set, vector, distance_type)])
    
    @classmethod
    def compute_closest_detector(cls, detector_set, vector, distance_type):
        distances = []
        if detector_set is not None and len(detector_set.detectors) > 0:
            for detector in detector_set.detectors:
                if detector.distance_type == distance_type == 'cosine':
                    #distance = np.linalg.norm(vector - detector.vector) - detector.radius
                    distance = fast_cosine_distance_with_radius(detector.vector, vector, detector.radius, 0) # TODO: be aware of detector radius tweak
                    #print(euclidean_distance(detector.vector, vector, detector.radius, 0), distance)
                elif detector.distance_type == distance_type == 'euclidean':
                    #distance = np.linalg.norm(vector - detector.vector) - detector.radius
                    #distance = euclidean_distance(detector.vector, vector, detector.radius, 0) # TODO: be aware of detector radius tweak
                    distance = euclidean_distance(detector.vector, vector, detector.radius * 0.75, 0) # TODO: be aware of detector radius tweak
                    #print('detector distances', detector.radius, distance, detector.vector[0], detector.vector[1], detector.vector[2], vector[0], vector[1], vector[2])
                distances.append(distance)
            max_distance = np.min(distances)  # maximum radius is set to the closest self
            max_index = np.argmin(distances)  # get the index of the max distance vector
            #print('Copute closest detector', euclidean_distance(detector_set.detectors[max_index].vector, vector, detector.radius, 0), max_distance) 
            return max_distance, detector_set.detectors[max_index].vector
        else:
            return 999999.0, None
        

    @classmethod
    def compute_closest_self(cls, self_df, self_region_radius, vector, distance_type):
        distances = []
        #vectors = self_df['vector']
        #distances = np.linalg.norm(vector - vectors, axis=1) - self_region_radius
        #print('type:', type(distances))
        #distances = 1 - fast_cosine_similarity([vector] * len(vectors), vectors) - self_region_radius
        #print('type:', type(distances))
        #print('Length of distances', len(distances))
        #distances = cosine_similarity([vector], vectorsnp.linalg.norm(vector - vectors, axis=1) - self_region_radius
        
        # check for distances to self samples
        for row in self_df.itertuples(index=False, name=None):
            if distance_type == 'cosine':
                distance = fast_cosine_distance_with_radius(vector, row[1], 0, self_region_radius)
            elif distance_type == 'euclidean':
                distance = euclidean_distance(vector, row[1], 0, self_region_radius)
            #if distance < 0:
            #    print('Negative distance:', distance, vector, row[1])
            distances.append(distance)
        # check for distances to other mature detectors
        #print('min before detectors', np.min(distances))

        #print('min after self:', np.min(distances))
        max_distance = np.min(distances)  # maximum radius is set to the closest self
        max_index = np.argmin(distances)  # get the index of the lowest distance
        #if euclidean_distance(self_df.iloc[min_index]['vector'], vector, 0, self_region_radius) != min_distance:
        #    print('NOT CORRECT VECTOR min index', min_index, 'min distance', min_distance, euclidean_distance(self_df.iloc[min_index]['vector'], vector, 0, self_region_radius))
        return max_distance, self_df.iloc[max_index]['vector']
        
    def to_dict(self):
        return {
            "vector": [float(v) for v in self.vector],
            "radius": float(self.radius),
            "distance_type": self.distance_type
        }

    @classmethod
    def from_dict(cls, data):
        return cls(np.array(data['vector'], dtype=np.float32), np.float32(data['radius']), data['distance_type'], None)
    
    def mutate(self, mutation_rate, change, max):
        indexes = list(range(len(self.vector)))
        mutation_indexes = random.sample(indexes, int(len(self.vector) * mutation_rate))

        for i in mutation_indexes:
            if random.randint(0,1) == 0 and self.vector[i] < max[i]:
                self.vector[i] += random.uniform(0, change[i])
            elif self.vector[i] > -max[i]:
                self.vector[i] -= random.uniform(0, change[i])
        #print('Mutation:', previous, self.vector)    
    
    def move_away_from_nearest(self, nearest_vector, step, feature_low, feature_max):
        direction = (self.vector - nearest_vector)    
        # Normalize the direction vector (optional, for consistent step size)
        #direction_normalized = direction / np.linalg.norm(direction)
        # Move in the opposite direction of b and c
        new_pos = self.vector + step * direction
        exceeding_max = new_pos > feature_max 
        exceeding_max_negative = new_pos < feature_low
        # only set new vector pos if none of the feature values exceed max
        if not (np.any(exceeding_max) or np.any(exceeding_max_negative)):
            self.vector = new_pos

    def compute_fitness(self, detector_set):
        area = np.pi * self.radius**2
        overlap = 0
        if detector_set is not None:
            for detector in detector_set.detectors:
                overlap += circle_overlap_area(self.vector, self.radius, detector.vector, detector.radius)
        self.fitness = area - overlap

class DetectorSet:
    def __init__(self, detectors):
        self.detectors = detectors
        '''self.grid = {}
        for detector in detectors:
            self.add_detector_to_grid(detector)'''

    '''def add_detector_to_grid(self, detector):
        grid_key = (int(detector.vector[0]), int(detector.vector[1]))  # discretize coordinates
        if grid_key not in self.grid:
            self.grid[grid_key] = []
        self.grid[grid_key].append(detector)  '''      

    '''def append_detector(self, detector):
        self.detectors.append(detector)
        self.add_detector_to_grid(detector)
        for grid_key, detectors in self.grid.items():
            print('Grid key:', grid_key, 'Detectors:', len(detectors))'''

    '''def negative_space_not_covered(self):
        for grid_key, detectors in self.grid.items():
            print('Grid key:', grid_key, 'Detectors:', len(detectors))
            area = 0
            overlap = 0
            for detector_a in detectors:
                area += np.pi * detector_a.radius**2
                for detector_b in detectors:
                    if detector_a != detector_b:                        
                        overlap = circle_overlap_area(detector_a.vector, detector_a.radius, detector_b.vector, detector_b.radius)
            print('Grid key:', grid_key, 'Area:', area, 'Overlap:', overlap, area - overlap)'''

    def to_dict(self):
        return {"detectors": [d.to_dict() for d in self.detectors]}

    @classmethod
    def from_dict(cls, data):
        return cls([Detector.from_dict(d) for d in data['detectors']])
    
    def save_to_file(self, filename):
        with open(f'model/detector/{filename}', 'w') as f:            
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, filename):
        with open(f'model/detector/{filename}', 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
class NegativeSelectionGeneticAlgorithm():
    def __init__(self, dim, pop_size, mutation_rate, self_region_rate, true_df, detector_set, distance_type='euclidean'):
        self.dim = dim
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
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
        #TODO: calculate the negative space based on total space minus self and their self regions
    
    def initiate_population(self):
        self.population = [Detector.create_detector(self.feature_low, self.feature_max, self.dim, self.true_df, self.self_region, self.detector_set, self.distance_type) for _ in range(self.pop_size)] 

    def find_distributions(self):
        random_detectors = [Detector.create_detector(self.feature_low, self.feature_max, self.dim, self.true_df, 0, None, self.distance_type) for _ in range(10)]
        cosine_distances = []
        euclidean_distances = []
        for vector in random_detectors:
            for row in self.true_df.itertuples(index=False, name=None):
                #distances.append(np.linalg.norm(vector.vector - row[1])) #self.vector_factory.document_vector(row[0]))
                cosine_distances.append(fast_cosine_distance(vector.vector, row[1]))
                euclidean_distances.append(euclidean_distance(vector.vector, row[1], 0, 0))
        return np.mean(euclidean_distances), np.std(euclidean_distances), np.mean(cosine_distances), np.std(cosine_distances)
            
    def compute_population_fitness(self):
        for detector in self.population:
            detector.compute_fitness(self.detector_set)

    # Currently averaging half of the indexes and using the remaining indexes from the other parent
    def recombine(self, parent1, parent2):
        '''indexes = list(range(self.dim))
        # draw which indexes should average
        indexes_to_average_for_offspring1 = random.sample(indexes, int(self.dim / 2))
        # remaining indexes from parent2
        indexes_to_average_for_offspring2 = list(set(indexes) - set(indexes_to_average_for_offspring1))
        offspring1 = np.zeros(self.dim, dtype=np.float32)
        offspring2 = np.zeros(self.dim, dtype=np.float32)
        for idx1, idx2 in zip(indexes_to_average_for_offspring1, indexes_to_average_for_offspring2):
            offspring1[idx1] = (parent1.vector[idx1] + parent2.vector[idx1]) / 2
            offspring1[idx2] = parent1.vector[idx2]
            offspring2[idx2] = (parent1.vector[idx2] + parent2.vector[idx2]) / 2
            offspring2[idx1] = parent2.vector[idx1]'''
        
        return [Detector((parent1.vector + parent2.vector) / 2, 0, self.distance_type, self.detector_set)] #, Detector(offspring2, 0, self.distance_type, self.detector_set)]

    def tournament_selection(self, tournament_size=2):
        # Randomly select tournament_size individuals and choose the best
        contenders = random.sample(self.population, tournament_size)
        return max(contenders, key=lambda individual: individual.fitness)

    def recombine_tournament(self):
        offspring_list = []
        
        # Perform tournament selection to select poulation size
        for _ in range(self.pop_size):
            # Select two individuals via tournament selection
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            #print(parent1.vector, parent2.vector)
            # Recombine the two parents to create offspring
            offsprings = self.recombine(parent1, parent2)
            
            # Add offsprings to the offspring list
            offspring_list.extend(offsprings)

            # Stop once we have offspring equal to half the population
            #if len(offspring_list) >= int(self.pop_size / 2):
            #    break

        return offspring_list

    def evolve_detector(self, stagnation_patience=10, pop_check_ratio=0.1, mutation_change_rate=0.005):
        # TODO: fix issue with detector being stuck(?) and concluding before reaching max radius by moving slightly
        self.initiate_population()
    
        best = 0
        stagnant = 0
        #mutation_change = mutation_change_rate * (self.feature_mean + self.feature_stdev * 3)
        #mutation_max = np.maxself.feature_mean + self.feature_stdev * 3
        #print('Mutation change', mutation_change, 'Mutation max', mutation_max)
        generations = 0
        while stagnant < stagnation_patience:
            # sort based on fitness
            self.population = sorted(self.population, key=lambda detector: detector.fitness, reverse=True)
            # recombine the best
            offsprings = self.recombine_tournament()
            # mutate
            for offspring in offsprings:
                # mutate (or move away from nearest)
                step = np.random.beta(a=1, b=4) # more likely to draw closer to 0
                #print('Step:', step)
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
                #offspring.mutate(self.mutation_rate, mutation_change, mutation_max)
                #distance_to_detector, nearest_detector = Detector.compute_closest_detector(self.detector_set, offspring.vector, self.distance_type)
                #distance_to_self, nearest_self = Detector.compute_closest_self(self.true_df, self.self_region, offspring.vector, self.distance_type)
                offspring.compute_fitness(self.detector_set)
            # bring mutated offspring into population

            self.population.extend(offsprings)
                   
            # add random detectors
            self.population.extend([Detector.create_detector(self.feature_low, self.feature_max, self.dim, self.true_df.sample(int(pop_check_ratio * len(self.true_df))), self.self_region, self.detector_set, self.distance_type) for _ in range(int(0.2 * self.pop_size))])
            #self.compute_population_fitness()
            # sort full population
            self.population = sorted(self.population, key=lambda detector: detector.fitness, reverse=True)           
        
            # select only the best to survive to the next generation
            self.population = self.population[:self.pop_size]                   
            
            #for i in range(len(self.population)):
            #    print(self.population[i].fitness, self.population[i].vector)

            # update stagnation (to decide on convergence)
            detector = self.population[0]
            if True: #generations % 10 == 0:
                print(detector.fitness, np.max(detector.vector), np.min(detector.vector))
            if abs(detector.fitness - best) > 0.01:
                best = detector.fitness
                stagnant = 0
            else:
                stagnant += 1
            generations += 1

        # TODO: below needs to be added if pop_check_ratio < 1
        if pop_check_ratio < 1:
            old_r = self.population[0].radius
            distance_to_detector, nearest_detector = Detector.compute_closest_detector(self.detector_set, offspring.vector, self.distance_type)
            distance_to_self, nearest_self = Detector.compute_closest_self(self.true_df, self.self_region, offspring.vector, self.distance_type)
            self.population[0].radius = np.min([distance_to_detector, distance_to_self])
            print(self.population[0].vector)
            self.population[0].compute_fitness(self.detector_set)
            print('Changed radius:', old_r, self.population[0].radius)

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
    
def test_distance(true_df, fake_df):
    distances = []
    for tv, fk in zip(true_df.sample(100)['vector'], fake_df.sample(100)['vector']):
        distances.append(euclidean_distance(tv, fk, 0, 0))
    return np.mean(distances), np.std(distances)

def main():
    a = np.array([1,3,3,1,1,1,1,1])
    b = np.array([1,2,3,1,1,1,1,1])
    d = np.array([1,1,1,1,1,1,1,1])
    print(euclidean_distance(a, d, 0, 0), euclidean_distance(b, d, 0, 0))

    dim = 10
    vec_factory = SpacyVectorFactory() # BERTVectorFactory() # SpacyVectorFactory() #BERTVectorFactory()
    true_df = vec_factory.load_vectorized_dataframe('dataset/ISOT/True_glove.h5')
    pca = PCA(n_components=dim)
    vectors_df = pd.DataFrame(true_df['vector'].tolist())
    reduced_vectors = pca.fit_transform(vectors_df)
    true_df['vector'] = [np.array(vec) for vec in reduced_vectors]
    print('PCA reduced', len(true_df.iloc[0]['vector']))
    true_training_df, tmp_df = train_test_split(true_df, test_size=0.4, random_state=42)
    true_validation_df, true_test_df = train_test_split(tmp_df, test_size=0.5, random_state=42)
    
    fake_df = vec_factory.load_vectorized_dataframe('dataset/ISOT/Fake_glove.h5')
    vectors_df = pd.DataFrame(fake_df['vector'].tolist())
    reduced_vectors = pca.transform(vectors_df)
    fake_df['vector'] = [np.array(vec) for vec in reduced_vectors]
    fake_training_df, tmp_df = train_test_split(fake_df, test_size=0.4, random_state=42)
    fake_validation_df, fake_test_df = train_test_split(tmp_df, test_size=0.5, random_state=42)

    print(test_distance(true_training_df, fake_training_df))

    '''d = [Detector.create_detector(0, 2, 300, true_training_df.sample(3), 0, None) for _ in range(5)]
    dset = DetectorSet([])  #DetectorSet([]) #DetectorSet.load_from_file('detectors_10.json')
    print(fast_cosine_distance_with_radius(d[0].vector, true_training_df.iloc[1].vector, 0.01, 0))
    d[0].radius = Detector.compute_maximum_radius(true_training_df.sample(500), 0.01, dset, d[0].vector, 0)
    print('radius', d[0].radius)
    nsga = NegativeSelectionGeneticAlgorithm(0, 2, 300, 20, 0.2, true_training_df, dset) 
    new_d = nsga.evolve_detector(20)
    print('new_d', new_d.radius)
    dset.detectors.append(new_d)
    new_d = nsga.evolve_detector(20)
    print('new d 2', new_d.radius)'''

    dset = DetectorSet([])  #DetectorSet([]) #DetectorSet.load_from_file('detectors_10.json')
    nsga = NegativeSelectionGeneticAlgorithm(3, 50, 0.2, true_training_df, dset, distance_type='cosine') 
    detector_size = 5
    for i in range(detector_size):
        detector = nsga.evolve_detector(1, pop_check_ratio=0.1, mutation_change_rate=0.01)
        if detector.fitness > 0:
            dset.detectors.append(detector)
        print('set length', len(dset.detectors))
        dset.save_to_file('detectors.json')
        print(len(dset.detectors))
        # Evaluate
        '''time0 = time.perf_counter()
        true_detected, true_total = nsga.detect(true_validation_df, dset, 9999)
        fake_detected, fake_total = nsga.detect(fake_validation_df, dset, 9999)
        print(time.perf_counter() - time0)
        print('Precision:', precision(fake_detected, true_detected), 'Recall', recall(fake_detected, fake_total - fake_detected))
        print('True/Real detected:', true_detected, 'Total real/true:', true_total, 'Fake detected:', fake_detected, 'Total fake:', fake_total)'''
    
    dset = DetectorSet.load_from_file('detectors.json')
    print('set length loaded', len(dset.detectors))
    time0 = time.perf_counter()
    true_detected, true_total = nsga.detect(true_validation_df, dset, 100)
    fake_detected, fake_total = nsga.detect(fake_validation_df, dset, 100)
    print(time.perf_counter() - time0)
    print('Precision:', precision(fake_detected, true_detected), 'Recall', recall(fake_detected, fake_total - fake_detected))
    print('True/Real detected:', true_detected, 'Total real/true:', true_total, 'Fake detected:', fake_detected, 'Total fake:', fake_total)

    '''
    d1 = Detector([0,1,2,6], radius=41)
    d2 = Detector([0,1,3,9], radius=44)
    dset = DetectorSet([d1, d2])
    dset.save_to_file('test_detectors.json')
    dset = DetectorSet.load_from_file('test_detectors.json')
    for d in [d1,d2]:
        print(d.vector)
    
    '''

if __name__ == "__main__":
    main()
