import pandas as pd
import numpy as np
import random
from vectorfactory import SpacyVectorFactory
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import time

class Detector():
    def __init__(self, mean, stdev, dim, true_df, vector=None):
        if vector is not None:
            self.vector = vector
        else:
            self.vector = np.random.normal(mean, stdev, dim)
            
        self.true_df = true_df
        self.update_minimum_distance_and_fitness() 
        self.compute_fitness()

    def mutate(self, mutation_rate):
        indexes = list(range(len(self.vector)))
        mutation_indexes = random.sample(indexes, int(len(self.vector) * mutation_rate))
        for i in mutation_indexes:
            if random.randint(0,1) == 0:
                self.vector[i] * (1 + mutation_rate)
            else:
                self.vector[i] * (1 - mutation_rate)
     
    def compute_fitness(self):
        self.fitness = self.distance_threshold

    def update_minimum_distance_and_fitness(self):
        distances = []
        for row in self.true_df.itertuples(index=False, name=None):
            distance = np.linalg.norm(self.vector - row[1]) #self.vector_factory.document_vector(row[0]))    
            distances.append(distance)
        self.distance_threshold = np.min(distances)
        self.fitness = self.distance_threshold
    
class NegativeSelectionGeneticAlgorithm():
    def __init__(self, mean, stdev, dim, pop_size, mutation_rate, true_df, fake_df):
        self.mean = mean
        self.stdev = stdev
        self.dim = dim
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.true_df = true_df
        self.fake_df = fake_df
        self.distance_mean, self.distance_std = self.extract_distance_distribution()
        self.self_region = 0.01 * self.distance_mean # use 1% of the mean distance between random detectors and self samples as self region (region around a self sample)
        
        self.population = [Detector(self.mean, self.stdev, self.dim, true_df) for _ in range(self.pop_size)]        
    
    def extract_distance_distribution(self):
        random_detectors = [Detector(self.mean, self.stdev, self.dim, self.true_df) for _ in range(10)]
        distances = []
        for vector in random_detectors:
            for row in self.true_df.itertuples(index=False, name=None):
                distances.append(np.linalg.norm(vector.vector - row[1])) #self.vector_factory.document_vector(row[0]))
        return np.mean(distances), np.std(distances)

    def evaluate_against_self(self, detector):
        for row in self.true_df.itertuples(index=False, name=None):
            distance = np.linalg.norm(detector.vector - row[1]) #self.vector_factory.document_vector(row[0]))
            #print(detector.vector, row[1])
            #print(distance, cosine_similarity([detector.vector], [row[1]])[0])
            
    def compute_population_fitness(self):
        for detector in self.population:
            detector.compute_fitness()

    def recombine(self, parent1, parent2):
        indexes = list(range(self.dim))
        # draw which indexes should be taken from first parent
        parent1_indexes = random.sample(indexes, int(self.dim / 2))
        # remaining indexes from parent2
        parent2_indexes = list(set(indexes) - set(parent1_indexes))
        offspring1 = np.zeros(self.dim)
        offspring2 = np.zeros(self.dim)
        for idx1, idx2 in zip(parent1_indexes, parent2_indexes):
            offspring1[idx1] = parent1.vector[idx1] # offspring1 get 1st set of values from parent1
            offspring1[idx2] = parent2.vector[idx2] # offspring1 get 2nd set of values from parent2
            offspring2[idx1] = parent1.vector[idx2] # offspring2 get 2nd set of values from parent1
            offspring2[idx2] = parent2.vector[idx1] # offspring2 get 1st set of values from parent2
        return [Detector(self.mean, self.stdev, self.dim, self.true_df, np.array(offspring1, dtype=np.float32)), Detector(self.mean, self.stdev, self.dim, self.true_df, np.array(offspring2, dtype=np.float32))]

    def recombine_best(self):
        best = self.population[:int(self.pop_size/2)]
        offspring_list = []
        for i in range(len(best)):
            offsprings = self.recombine(best[i], best[1+1])
            for offspring in offsprings:
                #offspring.update_minimum_distance_and_fitness()
                offspring_list.append(offspring)
        return offspring_list
        
    def evolve_detector(self):
        # sort based on fitness
        self.population = sorted(self.population, key=lambda detector: detector.fitness, reverse=True)
        # compute fitness 
        self.compute_population_fitness()
        # recombine the best
        offsprings = self.recombine_best()
        # mutate
        for offspring in offsprings:
            offspring.mutate(self.mutation_rate)
            offspring.update_minimum_distance_and_fitness()
        # bring mutated offspring into population
        self.population.extend(offsprings)
        # add random detectors
        self.population.extend([Detector(self.mean, self.stdev, self.dim, self.true_df) for _ in range(int(0.2 * self.pop_size))])
        # sort full population
        self.population = sorted(self.population, key=lambda detector: detector.fitness, reverse=True)
        # select only the best to survive to the next generation
        self.population = self.population[:self.pop_size]
        return self.population[0]
        

            


def main():
    vec_factory = SpacyVectorFactory()
    true_df = vec_factory.load_vectorized_dataframe('dataset/ISOT/True_vectorized.h5')
    true_training_df, tmp_df = train_test_split(true_df, test_size=0.4, random_state=42)
    true_validation_df, true_test_df = train_test_split(tmp_df, test_size=0.5, random_state=42)
    fake_df = vec_factory.load_vectorized_dataframe('dataset/ISOT/Fake_vectorized.h5')
    fake_training_df, tmp_df = train_test_split(fake_df, test_size=0.4, random_state=42)
    fake_validation_df, fake_test_df = train_test_split(tmp_df, test_size=0.5, random_state=42)
    nsga = NegativeSelectionGeneticAlgorithm(0, 2, 300, 20, 0.2, true_training_df, fake_training_df)
    best = 0
    stagnant = 0
    while stagnant < 10:
        detector = nsga.evolve_detector()
        if int(detector.fitness) > int(best):
            best = detector.fitness
            stagnant = 0
        else:
            if int(detector.fitness) == int(best):
                stagnant += 1
        print(detector.fitness, np.max(detector.vector), np.min(detector.vector))
        for row in true_df.itertuples(index=False, name=None):
            distance = np.linalg.norm(detector.vector - row[1]) #self.vector_factory.document_vector(row[0]))    
            if distance < detector.distance_threshold:
                print(distance, 'detected fake!')

    
    

if __name__ == "__main__":
    main()
