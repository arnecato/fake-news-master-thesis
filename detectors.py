import numpy as np
import random
import json
from util import euclidean_distance, fast_cosine_distance_with_radius

class Detector():
    def __init__(self, vector, radius, distance_type, fitness_function, feature_index=None):
        self.vector = vector
        self.radius = radius
        self.f1 = radius
        self.distance_type = distance_type    
        self.fitness_function = fitness_function
        self.distance = 0 # used for M.O crowding distance
        self.feature_index = feature_index

    def compute_fitness(self, *args, **kwargs):
        return self.fitness_function(self, *args, **kwargs)

    @classmethod
    def create_detector(cls, feature_low, feature_high, dim, self_df, self_region, detector_set, distance_type, fitness_function, feature_selection=0):
        feature_index = None
        if feature_selection > 0:
            feature_index = random.sample(range(len(feature_low)), feature_selection) 
            feature_low = feature_low[feature_index]
            feature_high = feature_high[feature_index]
        # randomly select a self sample to expand out from
        random_index = np.random.randint(0, len(self_df))
        self_sample_vector = self_df.iloc[random_index]['vector']
        self_sample_vector = np.copy(self_sample_vector) 
        #print('self vector', self_sample_vector)
        # randomly pick a feature index to expand out from and whether to go above or below the feature value
        feature_idx = random.choice(range(len(self_sample_vector)))
        #print(feature_idx, self_sample_vector)
        if random.choice([True, False]):
            feature_value = feature_high[feature_idx]
            self_sample_vector[feature_idx] = feature_value + random.uniform(0, 0.1 * feature_value)
        else:
            feature_value = feature_low[feature_idx]
            self_sample_vector[feature_idx] = feature_value - random.uniform(0, 0.1 * feature_value)
        #best_vector = np.random.uniform(low=feature_low, high=feature_high)
        best_vector = self_sample_vector
        #print('best vector', best_vector)
        best_distance, nearest_detector = Detector.compute_closest_detector(detector_set, best_vector, distance_type, feature_index)   
        #print('Creating new detector', mean, stdev, best_vector)
        #TODO: why do this 10 times? Why not just create one random detector?
        '''if detector_set is not None and len(detector_set.detectors) > 0:
            for _ in range(10):
                vector = np.random.uniform(low=feature_low, high=feature_high)
                distance_to_detector, nearest_detector = Detector.compute_closest_detector(detector_set, vector, distance_type, feature_index)
                if distance_to_detector > best_distance:
                    best_distance = distance_to_detector
                    best_vector = vector   
                    #print('new vector', best_distance, vector)'''
        detector = Detector(best_vector, 0, distance_type, fitness_function, feature_index)
        #distance_to_detector, nearest_detector = Detector.compute_closest_detector(detector_set, best_vector, distance_type, feature_index)
        distance_to_self, nearest_self = Detector.compute_closest_self(self_df, self_region, best_vector, distance_type, feature_index)
        detector.radius = distance_to_self #TODO: completely changed the logic here. Need to re-check!
        #detector.radius = np.min([distance_to_detector, distance_to_self]) #TODO: completely changed the logic here. Need to re-check!
        #print('created new', detector.radius, detector.vector)
        #detector.compute_fitness(detector_set)    
   
        return detector
    
    @classmethod
    def compute_closest_detector(cls, detector_set, vector, distance_type, feature_index):
        distances = []
        if detector_set is not None and len(detector_set.detectors) > 0:
            for detector in detector_set.detectors:
                # make sure detectors share the same feature indices
                if feature_index is not None:
                    detector_vector = detector.vector[feature_index]
                else:
                    detector_vector = detector.vector
                # only compare if they share 1 or more features
                if len(detector_vector) > 0:
                    if detector.distance_type == distance_type == 'cosine':
                        #distance = np.linalg.norm(vector - detector.vector) - detector.radius
                        distance = fast_cosine_distance_with_radius(detector.vector, vector, detector.radius, 0) # TODO: be aware of detector radius tweak
                        #print(euclidean_distance(detector.vector, vector, detector.radius, 0), distance)
                    elif detector.distance_type == distance_type == 'euclidean':
                        #distance = np.linalg.norm(vector - detector.vector) - detector.radius
                        #distance = euclidean_distance(detector.vector, vector, detector.radius, 0) # TODO: be aware of detector radius tweak
                        distance = euclidean_distance(detector.vector, vector, detector.radius * 1, 0) # TODO: be aware of detector radius tweak
                        #print('detector distances', detector.radius, distance, detector.vector[0], detector.vector[1], detector.vector[2], vector[0], vector[1], vector[2])
                    distances.append(distance)
                # they share no features, so distance is set to infinity
                else:
                    distances.append(float('inf'))
            max_distance = np.min(distances)  # maximum radius is set to the closest self
            max_index = np.argmin(distances)  # get the index of the max distance vector
            #print('Copute closest detector', euclidean_distance(detector_set.detectors[max_index].vector, vector, detector.radius, 0), max_distance) 
            return max_distance, detector_set.detectors[max_index].vector
        else:
            return float('inf'), None
        
    @classmethod
    def compute_closest_self(cls, self_df, self_region_radius, vector, distance_type, feature_index):
        distances = []
        
        # check for distances to self samples
        for row in self_df.itertuples(index=False, name=None):
            self_vector = row[1]
            if feature_index is not None:
                self_vector = row[1][feature_index]
            
            if distance_type == 'cosine':
                distance = fast_cosine_distance_with_radius(vector, self_vector, 0, self_region_radius)
            elif distance_type == 'euclidean':
                distance = euclidean_distance(vector, self_vector, 0, self_region_radius)
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
        d = {
            "vector": [float(v) for v in self.vector],
            "radius": float(self.radius),
            "distance_type": self.distance_type,
        }
        if self.feature_index is not None:
            d['feature_index'] = self.feature_index
        return d

    @classmethod
    def from_dict(cls, data, fitness_function):
        feature_index = data.get('feature_index')
        if feature_index is not None:
            return cls(np.array(data['vector'], dtype=np.float32), np.float32(data['radius']), data['distance_type'], fitness_function, feature_index)
        else:
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
    