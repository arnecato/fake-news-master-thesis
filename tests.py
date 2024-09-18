import unittest
from ga_nsa import main
from util import fast_cosine_similarity
import numpy as np
import pandas as pd
from ga_nsa import NegativeSelectionGeneticAlgorithm, DetectorSet, Detector

def test_fast_cosine_similarity_identical_vectors():
    a = np.array([1, 2, 3])
    b = np.array([1, 2, 3])
    assert fast_cosine_similarity(a, b) == 1.0, "Cosine similarity of identical vectors should be 1"

def test_fast_cosine_similarity_orthogonal_vectors():
    a = np.array([1, 0, 0])
    b = np.array([0, 1, 0])
    assert fast_cosine_similarity(a, b) == 0.0, "Cosine similarity of orthogonal vectors should be 0"

def test_fast_cosine_similarity_opposite_vectors():
    a = np.array([1, 2, 3])
    b = np.array([-1, -2, -3])
    assert fast_cosine_similarity(a, b) == -1.0, "Cosine similarity of opposite vectors should be -1"

def test_fast_cosine_similarity_random_vectors():
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    expected_similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    assert np.isclose(fast_cosine_similarity(a, b), expected_similarity), "Cosine similarity of random vectors is incorrect"

class TestFastCosineSimilarity(unittest.TestCase):
    def test_identical_vectors(self):
        test_fast_cosine_similarity_identical_vectors()

    def test_orthogonal_vectors(self):
        test_fast_cosine_similarity_orthogonal_vectors()

    def test_opposite_vectors(self):
        test_fast_cosine_similarity_opposite_vectors()

    def test_random_vectors(self):
        test_fast_cosine_similarity_random_vectors()

class TestComputeClosestDetector(unittest.TestCase):

    def setUp(self):
        self.detector1 = Detector(np.array([1.0, 2.0, 3.0]), 0.5, 'euclidean')
        self.detector2 = Detector(np.array([4.0, 5.0, 6.0]), 0.5, 'euclidean')
        self.detector_set = DetectorSet([self.detector1, self.detector2])
        self.test_vector = np.array([1.5, 2.5, 3.5])

    def test_compute_closest_detector_euclidean(self):
        distance, closest_vector = Detector.compute_closest_detector(self.detector_set, self.test_vector, 'euclidean')
        self.assertAlmostEqual(distance, 0.8660254037844386, places=5)
        np.testing.assert_array_almost_equal(closest_vector, self.detector1.vector)

    def test_compute_closest_detector_cosine(self):
        self.detector1.distance_type = 'cosine'
        self.detector2.distance_type = 'cosine'
        distance, closest_vector = Detector.compute_closest_detector(self.detector_set, self.test_vector, 'cosine')
        self.assertAlmostEqual(distance, 0.004504527, places=5)
        np.testing.assert_array_almost_equal(closest_vector, self.detector1.vector)

    def test_compute_closest_detector_empty_set(self):
        empty_detector_set = DetectorSet([])
        distance, closest_vector = Detector.compute_closest_detector(empty_detector_set, self.test_vector, 'euclidean')
        self.assertEqual(distance, 999999.0)
        self.assertIsNone(closest_vector)

if __name__ == '__main__':
    unittest.main()

    def test_compute_maximum_radius(self):
        radius = Detector.compute_closest_self(
            self.self_df, self.self_region_radius, self.vector, 'euclidean'
        )
        self.assertIsInstance(radius, float)
        self.assertGreaterEqual(radius, 0)

    def test_compute_maximum_radius_no_detectors(self):
        radius = Detector.compute_closest_self(
            self.self_df, self.self_region_radius, self.vector, 'euclidean'
        )
        self.assertIsInstance(radius, float)
        self.assertGreaterEqual(radius, 0)

    def test_compute_maximum_radius_empty_self_df(self):
        empty_self_df = pd.DataFrame({'vector': []})
        radius = Detector.compute_closest_self(
            empty_self_df, self.self_region_radius, self.vector, 'euclidean'
        )
        self.assertIsInstance(radius, float)
        self.assertGreaterEqual(radius, 0)

class TestNegativeSelectionGeneticAlgorithm(unittest.TestCase):
    def setUp(self):
        self.mean = 0
        self.stdev = 1
        self.dim = 10
        self.pop_size = 20
        self.mutation_rate = 0.1
        self.true_df = pd.DataFrame({
            'text': ['' for _ in range(10)],
            'vector': [np.random.normal(0, 2, self.dim) for _ in range(10)]
        })
        self.detector_set = None  # Assuming detector_set is not needed for recombine tests
        self.nsga = NegativeSelectionGeneticAlgorithm(self.mean, self.stdev, self.dim, self.pop_size, self.mutation_rate, self.true_df, self.detector_set)
        
        # Create two parent detectors
        self.parent1 = Detector(np.random.normal(self.mean, self.stdev, self.dim), 0)
        self.parent2 = Detector(np.random.normal(self.mean, self.stdev, self.dim), 0)

    def test_recombine(self):
        mean = 0
        stdev = 1
        dim = 4
        pop_size = 10
        mutation_rate = 0.1
        true_df = None
        detector_set = None

        nsga = NegativeSelectionGeneticAlgorithm(mean, stdev, dim, pop_size, mutation_rate, true_df, detector_set)

        parent1 = Detector(np.random.normal(mean, stdev, dim), 1.0)
        parent2 = Detector(np.random.normal(mean, stdev, dim), 1.0)

        offsprings = nsga.recombine(parent1, parent2)

        assert len(offsprings) == 2, "Recombine should produce two offsprings"
        assert len(offsprings[0].vector) == dim, "Offspring vector should have the same dimension as parents"
        assert len(offsprings[1].vector) == dim, "Offspring vector should have the same dimension as parents"

        # Check that half of the values come from one parent and the other half from the other parent
        for offspring in offsprings:
            #print('Offspring:\n', offspring.vector)
            #print(parent1.vector, '\n', parent2.vector)
            parent1_count = sum(1 for i in range(dim) if np.isclose(offspring.vector[i], parent1.vector[i]))
            parent2_count = sum(1 for i in range(dim) if np.isclose(offspring.vector[i], parent2.vector[i]))
            assert parent1_count == dim // 2, f"Half of the offspring vector should come from parent1: {parent1_count}"
            assert parent2_count == dim // 2, f"Half of the offspring vector should come from parent2: {parent2_count}"
            

'''class TestMainFunction(unittest.TestCase):
    def test_main_execution(self):
        try:
            main()
        except Exception as e:
            self.fail(f"main() raised an exception {e}")'''

if __name__ == "__main__":
    unittest.main()