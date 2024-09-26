import numpy as np
import random
import json
from util import euclidean_distance, fast_cosine_distance_with_radius

class Detector():
    def __init__(self, vector, radius, distance_type, fitness_function):
        self.vector = vector
        self.radius = radius
        self.f1 = radius
        self.distance_type = distance_type    
        self.fitness_function = fitness_function
        self.distance = 0 # used for M.O crowding distance

    def compute_fitness(self, *args, **kwargs):
        return self.fitness_function(self, *args, **kwargs)

    @classmethod
    def create_detector(cls, feature_low, feature_high, dim, self_df, self_region, detector_set, distance_type, fitness_function):
        best_vector = np.random.uniform(low=feature_low, high=feature_high)

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
        detector = Detector(best_vector, 0, distance_type, fitness_function)
        distance_to_detector, nearest_detector = Detector.compute_closest_detector(detector_set, best_vector, distance_type)
        distance_to_self, nearest_self = Detector.compute_closest_self(self_df, self_region, best_vector, distance_type)
        
        detector.radius = np.min([distance_to_detector, distance_to_self]) 
        #print('created new', detector.radius, detector.vector)
        #detector.compute_fitness(detector_set)    
   
        return detector
    
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
    def from_dict(cls, data, fitness_function):
        return cls(np.array(data['vector'], dtype=np.float32), np.float32(data['radius']), data['distance_type'], fitness_function)
    
    '''def mutate(self, mutation_rate, change, max):
        indexes = list(range(len(self.vector)))
        mutation_indexes = random.sample(indexes, int(len(self.vector) * mutation_rate))

        for i in mutation_indexes:
            if random.randint(0,1) == 0 and self.vector[i] < max[i]:
                self.vector[i] += random.uniform(0, change[i])
            elif self.vector[i] > -max[i]:
                self.vector[i] -= random.uniform(0, change[i])'''
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

    # Currently averaging the position of the two parents
    def recombine(self, other_parent, fitness_function):        
        return [Detector((self.vector + other_parent.vector) / 2, 0, self.distance_type, fitness_function)] #, Detector(offspring2, 0, self.distance_type, self.detector_set)]

    def dominated_by(self, comp_individual):
        ''' checks if this individual is dominated by the comp_individual '''
        if (comp_individual.f1 > self.f1 and comp_individual.f2 <= self.f2) or (comp_individual.f1 >= self.f1 and comp_individual.f2 < self.f2):
            return True
        else:
            return False
class DetectorSet:
    def __init__(self, detectors):
        self.detectors = detectors

    def to_dict(self):
        return {"detectors": [d.to_dict() for d in self.detectors]}

    @classmethod
    def from_dict(cls, data, fitness_function):
        return cls([Detector.from_dict(d, fitness_function) for d in data['detectors']])
    
    def save_to_file(self, filename):
        with open(f'{filename}', 'w') as f:            
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, filename, fitness_function):
        with open(f'{filename}', 'r') as f:
            data = json.load(f)
        return cls.from_dict(data, fitness_function)
    