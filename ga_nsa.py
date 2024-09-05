import pandas as pd
import numpy as np
import random
from vectorfactory import SpacyVectorFactory
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import time
import json

class Detector():
    def __init__(self, vector, radius):
        self.vector = vector
        self.radius = radius
        self.fitness = radius

    @classmethod
    def create_detector(cls, mean, stdev, dim, self_df, self_region, detector_set):
        vector = np.random.normal(mean, stdev, dim)
        radius = Detector.compute_maximum_radius(self_df, self_region, detector_set, vector)
        return Detector(vector, radius)

    @classmethod
    def compute_maximum_radius(cls, self_df, self_region_radius, detector_set, vector):
        distances = []
        # check for distances to self
        for row in self_df.itertuples(index=False, name=None):
            distance = np.linalg.norm(vector - row[1]) - self_region_radius
            distances.append(distance)
        # check for distances to other mature detectors
        if detector_set is not None:
            for detector in detector_set.detectors:
                distance = np.linalg.norm(vector - detector.vector) - detector.radius
        return np.min(distances) # maximum radius is set to the closest self
    
    def to_dict(self):
        return {
            "vector": [float(v) for v in self.vector],
            "radius": float(self.radius)
        }

    @classmethod
    def from_dict(cls, data):
        return cls(np.array(data['vector'], dtype=np.float32), np.float32(data['radius']))
    
    def mutate(self, mutation_rate):
        indexes = list(range(len(self.vector)))
        mutation_indexes = random.sample(indexes, int(len(self.vector) * mutation_rate))
        for i in mutation_indexes:
            if random.randint(0,1) == 0:
                self.vector[i] * (1 + mutation_rate)
            else:
                self.vector[i] * (1 - mutation_rate)
     
    def compute_fitness(self):
        self.fitness = self.radius

class DetectorSet:
    def __init__(self, detectors):
        self.detectors = detectors

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
    def __init__(self, mean, stdev, dim, pop_size, mutation_rate, true_df, fake_df):
        self.mean = mean
        self.stdev = stdev
        self.dim = dim
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.true_df = true_df
        self.fake_df = fake_df
        self.detector_set = DetectorSet([])
        self.distance_mean, self.distance_std = self.extract_distance_distribution()
        self.self_region = 0.01 * self.distance_mean # use 1% of the mean distance between random detectors and self samples as self region (region around a self sample)
        
    
    def initiate_population(self):
        self.population = [Detector.create_detector(self.mean, self.stdev, self.dim, self.true_df, self.self_region, self.detector_set) for _ in range(self.pop_size)] 

    def extract_distance_distribution(self):
        random_detectors = [Detector.create_detector(self.mean, self.stdev, self.dim, self.true_df, 0, None) for _ in range(10)]
        distances = []
        for vector in random_detectors:
            for row in self.true_df.itertuples(index=False, name=None):
                distances.append(np.linalg.norm(vector.vector - row[1])) #self.vector_factory.document_vector(row[0]))
        return np.mean(distances), np.std(distances)
            
    def compute_population_fitness(self):
        for detector in self.population:
            detector.compute_fitness()

    def recombine(self, parent1, parent2):
        indexes = list(range(self.dim))
        # draw which indexes should be taken from first parent
        parent1_indexes = random.sample(indexes, int(self.dim / 2))
        # remaining indexes from parent2
        parent2_indexes = list(set(indexes) - set(parent1_indexes))
        offspring1 = np.zeros(self.dim, dtype=np.float32)
        offspring2 = np.zeros(self.dim, dtype=np.float32)
        for idx1, idx2 in zip(parent1_indexes, parent2_indexes):
            offspring1[idx1] = parent1.vector[idx1] # offspring1 get 1st set of values from parent1
            offspring1[idx2] = parent2.vector[idx2] # offspring1 get 2nd set of values from parent2
            offspring2[idx1] = parent1.vector[idx2] # offspring2 get 2nd set of values from parent1
            offspring2[idx2] = parent2.vector[idx1] # offspring2 get 1st set of values from parent2
        return [Detector(offspring1, Detector.compute_maximum_radius(self.true_df, self.self_region, self.detector_set, offspring1)),
                Detector(offspring2, Detector.compute_maximum_radius(self.true_df, self.self_region, self.detector_set, offspring2))]

    def recombine_best(self):
        best = self.population[:int(self.pop_size/2)]
        offspring_list = []
        for i in range(len(best)):
            offsprings = self.recombine(best[i], best[1+1])
            for offspring in offsprings:
                offspring_list.append(offspring)
        return offspring_list
        
    def detect(self, df, detector_set):
        detected = 0
        total = 0
        for row in df.itertuples(index=False, name=None):
            for detector in detector_set.detectors:
                distance = np.linalg.norm(detector.vector - row[1]) #self.vector_factory.document_vector(row[0]))    
                #print(distance, detector.distance_threshold)
                if distance < detector.radius:
                    print(distance, 'detected!')
                    detected += 1
                    break
            total += 1             
        return detected, total

    def evolve_detector(self, stagnation_patience=10):
        self.initiate_population()
        best = 0
        stagnant = 0
        while stagnant < stagnation_patience:
            # sort based on fitness
            self.population = sorted(self.population, key=lambda detector: detector.fitness, reverse=True)
            # compute fitness 
            self.compute_population_fitness()
            # recombine the best
            offsprings = self.recombine_best()
            # mutate
            for offspring in offsprings:
                offspring.mutate(self.mutation_rate)
                offspring.radius = Detector.compute_maximum_radius(self.true_df, self.self_region, self.detector_set, offspring.vector)
                offspring.fitness = offspring.radius
            # bring mutated offspring into population
            self.population.extend(offsprings)
            # add random detectors
            self.population.extend([Detector.create_detector(self.mean, self.stdev, self.dim, self.true_df, self.self_region, self.detector_set) for _ in range(int(0.2 * self.pop_size))])
            # sort full population
            self.population = sorted(self.population, key=lambda detector: detector.fitness, reverse=True)
            # select only the best to survive to the next generation
            self.population = self.population[:self.pop_size]

            # update stagnation (to decide on convergence)
            detector = self.population[0]
            print(detector.fitness, np.max(detector.vector), np.min(detector.vector))
            if int(detector.fitness) > int(best):
                best = detector.fitness
                stagnant = 0
            else:
                if int(detector.fitness) == int(best):
                    stagnant += 1

        return self.population[0]
        

            


def main():
    vec_factory = SpacyVectorFactory()
    true_df = vec_factory.load_vectorized_dataframe('dataset/ISOT/True_vectorized.h5')
    true_training_df, tmp_df = train_test_split(true_df, test_size=0.4, random_state=42)
    true_validation_df, true_test_df = train_test_split(tmp_df, test_size=0.5, random_state=42)
    fake_df = vec_factory.load_vectorized_dataframe('dataset/ISOT/Fake_vectorized.h5')
    fake_training_df, tmp_df = train_test_split(fake_df, test_size=0.4, random_state=42)
    fake_validation_df, fake_test_df = train_test_split(tmp_df, test_size=0.5, random_state=42)
    nsga = NegativeSelectionGeneticAlgorithm(0, 2, 300, 10, 0.2, true_training_df, fake_training_df)    
    detectors = []
    for i in range(100):
        detector = nsga.evolve_detector(5)
        detectors.append(detector)
        dset = DetectorSet(detectors)
        dset.save_to_file('detectors_100.json')
    
    dset = DetectorSet.load_from_file('detectors_100.json')
    detected, total = nsga.detect(true_test_df, dset)
    print(detected, total, 'Recall:', detected / total)

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
