import numpy as np
import random
import json
from util import euclidean_distance, fast_cosine_distance_with_radius, get_shared_feature_vectors, get_nearby_self

class Detector():
    def __init__(self, vector, radius, distance_type, fitness_function):
        #print('feature index', feature_index)
        self.vector = vector
        self.radius = radius
        self.f1 = radius
        self.distance_type = distance_type    
        self.fitness_function = fitness_function
        self.distance = 0 # used for M.O crowding distance
        #self.feature_index = feature_index

    def compute_fitness(self, *args, **kwargs):
        return self.fitness_function(self, *args, **kwargs)

    @classmethod
    def create_detector(cls, feature_low, feature_high, dim, self_df, self_region, detector_set, distance_type, fitness_function, self_range):
        '''feature_index = list(range(len(self_df.iloc[0]['vector'])))
        if feature_selection > 0:
            #print('Feature selection:', feature_selection)
            feature_index = random.sample(range(len(feature_low)), feature_selection) 
            feature_low = feature_low[feature_index]
            feature_high = feature_high[feature_index]
        # randomly select a self sample to expand out from
        random_index = np.random.randint(0, len(self_df))
        self_sample_vector = self_df.iloc[random_index]['vector'][feature_index]'''
        
        self_sample_vector = np.random.uniform(low=feature_low, high=feature_high)
        #print('self vector', self_sample_vector)
        
        # half of the time, it should be a random vector drawn uniformly from the feature space
        '''if random.choice([True, False]) or voronoi_points is None:
            self_sample_vector = np.random.uniform(low=feature_low, high=feature_high)
        # draw from voronoi points
        else:
            self_sample_vector = random.choice(voronoi_points)'''
        '''# the other half of the time a self sample is selected randomly and a detector is initialized just outside of it
        else:
            # randomly pick a feature index to expand out from and whether to go above or below the feature value
            feature_idx = random.choice(range(len(self_sample_vector)))
            #print(feature_idx, self_sample_vector)
            if random.choice([True, False]):
                feature_value = self_sample_vector[feature_idx]
                self_sample_vector[feature_idx] = feature_value + self_region + random.uniform(0, self_region)
            else:
                feature_value = self_sample_vector[feature_idx]
                self_sample_vector[feature_idx] = feature_value - self_region - random.uniform(0, self_region)'''
        best_vector = self_sample_vector
        exceeding_max = best_vector > feature_high + abs(feature_high * 0.1)
        exceeding_max_negative = best_vector < feature_low - abs(feature_low * 0.1)
        if np.any(exceeding_max) or np.any(exceeding_max_negative):
            print('new detect pos is over!', best_vector, feature_high + abs(feature_high * 0.1), feature_low - abs(feature_low * 0.1))
            best_vector = np.clip(best_vector, feature_low, feature_high)
        #print('best vector', best_vector)
        #best_distance, nearest_detector = Detector.compute_closest_detector(detector_set, best_vector, distance_type, feature_index)   #TODO: remove this?
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
        detector = Detector(best_vector, 0, distance_type, fitness_function)
        distance_to_detector, nearest_detector = Detector.compute_closest_detector(detector_set, best_vector, distance_type)
        median_distance_to_self, nearest_self, closest_selves = Detector.compute_closest_self(self_df, self_region, best_vector, distance_type, self_range)
        detector.radius = np.min([distance_to_detector, median_distance_to_self]) 
        #print('created new', detector.radius, detector.vector)
        #detector.compute_fitness(detector_set)    
   
        return detector
    
    @classmethod
    def compute_closest_detector(cls, detector_set, vector, distance_type):
        distances = []
        if detector_set is not None and len(detector_set.detectors) > 0:
            for detector in detector_set.detectors:
                # make sure detectors share the same feature indices
                #if feature_index is not None or detector.feature_index is not None:
                #    #print(detector.feature_index, feature_index)
                #    vector, detector_vector, shared_features = get_shared_feature_vectors(vector, feature_index, detector.vector, detector.feature_index)
                #detector_vector = detector.vector
                # only compare if they share 1 or more features
                detector_vector = detector.vector
                if vector is not None and detector_vector is not None and len(detector_vector) > 0:
                    if detector.distance_type == distance_type == 'cosine':
                        #distance = np.linalg.norm(vector - detector.vector) - detector.radius
                        distance = fast_cosine_distance_with_radius(detector_vector, vector, 0, 0) # TODO: be aware of detector radius tweak
                        #print(euclidean_distance(detector.vector, vector, detector.radius, 0), distance)
                    elif detector.distance_type == distance_type == 'euclidean':
                        #distance = np.linalg.norm(vector - detector.vector) - detector.radius
                        #distance = euclidean_distance(detector.vector, vector, detector.radius, 0) # TODO: be aware of detector radius tweak
                        distance = euclidean_distance(detector_vector, vector, 0, 0) # TODO: be aware of detector radius tweak
                        #print('detector distances', detector.radius, distance, detector.vector[0], detector.vector[1], detector.vector[2], vector[0], vector[1], vector[2])
                    distances.append(distance)
                # they share no features, so distance is set to infinity
                else:
                    distances.append(float('inf'))
            max_distance = np.min(distances)  # maximum radius is set to the closest self
            max_index = np.argmin(distances)  # get the index of the max distance vector
            #print('Copute closest detector', euclidean_distance(detector_set.detectors[max_index].vector, vector, detector.radius, 0), max_distance) 
            return max_distance, detector_set.detectors[max_index] #.vector #TODO: prehaps change to object instead of vector on the compute_closest_self too, like this
        else:
            return float('inf'), None
        
    @classmethod
    def compute_closest_self(cls, self_points, self_region_radius, vector, distance_type, feature_range):
        distances = []
        closest_selves = []    
        closest_distance = float('inf')

        #print('range', np.max(feature_range) * 0.3)
        for i in np.arange(0.01, 1.5, 0.1):
            #print(self_points)
            nearby_self = get_nearby_self(self_points, vector, np.max(feature_range) * i)
            if len(nearby_self) > 1:
                break
        if len(nearby_self) == 0:
            raise Exception(f'No nearby self points found {vector}, {feature_range}')
        
        for self_vector in nearby_self: #TODO: double check that this type of value always makes sense
            #self_vector = self_vector[1]
            #if feature_index is not None:
            #    self_vector = row[1][feature_index]
            
            if distance_type == 'cosine':
                distance = fast_cosine_distance_with_radius(vector, self_vector, 0, self_region_radius)
            elif distance_type == 'euclidean':
                distance = euclidean_distance(vector, self_vector, 0, self_region_radius)
            #if distance < 0:
            #    print('Negative distance:', distance, vector, row[1])        
            distances.append(distance)
            if distance < closest_distance:
                closest_selves.append(self_vector)
        # check for distances to other mature detectors
        #print('min before detectors', np.min(distances))

        #print('min after self:', np.min(distances))
        min_distance = np.min(distances)  # maximum radius is set to the closest self
        #if len(vector) == 2:
        #    print('max distance', max_distance)
        min_index = np.argmin(distances)  # get the index of the lowest distance

        # TODO: recheck this code 
        # take the median of the closest selves!
        distances[min_index] = float('inf')
        min_distance_next = np.min(distances)
        distances[min_index] = min_distance
        median_distance = np.median([min_distance, min_distance_next])
        #if euclidean_distance(self_df.iloc[min_index]['vector'], vector, 0, self_region_radius) != min_distance:
        #    print('NOT CORRECT VECTOR min index', min_index, 'min distance', min_distance, euclidean_distance(self_df.iloc[min_index]['vector'], vector, 0, self_region_radius))
        return min_distance, self_points[min_index], closest_selves # ['vector']
        
    def to_dict(self):
        d = {
            "vector": [float(v) for v in self.vector],
            "radius": float(self.radius)
        }
        #if self.feature_index is not None:
        #    d['feature_index'] = self.feature_index
        #print('feature index', self.feature_index)
        return d

    @classmethod
    def from_dict(cls, data, fitness_function):
        return cls(np.array(data['vector'], dtype=np.float32), np.float32(data['radius']), 'euclidean', fitness_function)
    
    def mutate(self, feature_low, feature_max):
        old_vector = self.vector.copy()
        for i in range(len(self.vector)):
            range_i = feature_max[i] - feature_low[i]
            mutation_step = np.random.beta(a=1, b=4) * 0.1 * range_i
            self.vector[i] += mutation_step if random.choice([True, False]) else -mutation_step
            self.vector[i] = np.clip(self.vector[i], feature_low[i], feature_max[i])
            #print('Old and new vector', old_vector, self.vector, self.vector - old_vector, mutation_step)
    
    def move_away_from_nearest(self, nearest_vector, step, feature_low, feature_max):
        #print('move away from nearest', self.vector, nearest_vector)
        direction = (nearest_vector - self.vector)    
        # Normalize the direction vector (optional, for consistent step size)
        #direction_normalized = direction / np.linalg.norm(direction)
        # Move in the opposite direction of b and c
        new_pos = self.vector - step * direction
        #print('new pos', new_pos, 'feature_max', feature_max, 'feature_low', feature_low)
        exceeding_max = new_pos > feature_max 
        exceeding_max_negative = new_pos < feature_low
        # only set new vector pos if none of the feature values exceed max
        if not (np.any(exceeding_max) or np.any(exceeding_max_negative)):
            self.vector = new_pos

    # Currently averaging the position of the two parents
    def recombine(self, other_parent, fitness_function):    
        #common_features = list(set(self.feature_index).intersection(other_parent.feature_index))
        '''if self.feature_index is None or other_parent.feature_index is None:
            feature_index = list(range(len(self.vector)))
        else:
            feature_index = sorted(set(self.feature_index + other_parent.feature_index))
        new_vector = np.zeros(len(feature_index))
        for i, idx in enumerate(feature_index):
            if idx in self.feature_index and idx in other_parent.feature_index:
                new_vector[i] = (self.vector[self.feature_index.index(idx)] + other_parent.vector[other_parent.feature_index.index(idx)]) / 2
            elif idx in self.feature_index:
                new_vector[i] = self.vector[self.feature_index.index(idx)]
            else:
                new_vector[i] = other_parent.vector[other_parent.feature_index.index(idx)]'''
        #if len(feature_index) != len(new_vector):
        #    print('feature index **********************', feature_index, new_vector)

        new_vector = (self.vector + other_parent.vector) / 2
        return [Detector(new_vector, 0, self.distance_type, fitness_function)] #, Detector(offspring2, 0, self.distance_type, self.detector_set)]

    def dominated_by(self, comp_individual):
        ''' checks if this individual is dominated by the comp_individual '''
        if (comp_individual.f1 > self.f1 and comp_individual.f2 >= self.f2) or (comp_individual.f1 >= self.f1 and comp_individual.f2 > self.f2):
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
    