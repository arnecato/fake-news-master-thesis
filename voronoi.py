import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from detectors import DetectorSet
from nsgaii_nsa import compute_fitness
import time

dset = DetectorSet.load_from_file("model/detector/detectors_bert_2dim_3600_6000_no_detector_radius.json", compute_fitness)
print('Detectors:', len(dset.detectors))
feature_index_lengths = {}
detector_points = []

for i in range(len(dset.detectors)-1,len(dset.detectors)):
    start_time = time.time()
    for detector in dset.detectors[:i]:
        detector_vector = detector.vector
        # Assuming detector_vector contains the coordinates (x, y)
        print(detector_vector)
        detector_points.append(np.array(detector_vector))
    
    # Convert list to numpy array
    detector_points = detector_points

    # Create the Voronoi diagram using detector centers
    vor = Voronoi(detector_points)
    end_time = time.time()
    print(f"Time taken for iteration {i}: {end_time - start_time} seconds")
    # Plot the Voronoi diagram
    fig, ax = plt.subplots()
    voronoi_plot_2d(vor, ax=ax)
    # Print the Voronoi points
    print("Voronoi vertices:")
    print(vor.vertices)
    # Customize plot (set limits based on your data)
    #ax.set_xlim([min_x, max_x])
    #ax.set_ylim([min_y, max_y])
    plt.show()