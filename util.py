import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import math

# Create 2D visualization
def visualize_2d(true_df, fake_df, detector_set, self_region):
    detector_positions = np.array([detector.vector for detector in detector_set.detectors])
    true_cluster = np.array(true_df['vector'].tolist())
    fake_cluster = np.array(fake_df['vector'].tolist())
    plt.scatter(true_cluster[:, 0], true_cluster[:, 1], color='blue', label='True', alpha=0.25)
    plt.scatter(fake_cluster[:, 0], fake_cluster[:, 1], color='red', label='Fake', alpha=0.25)
    #random_points = np.random.normal(0, 0.78, size=(200, 2))
    #plt.scatter(random_points[:, 0], random_points[:, 1], color='purple', label='Random Points')
    ax = plt.gca()
    ax.set_aspect('equal')
    for detector in detector_set.detectors:
        detector_circle = plt.Circle(detector.vector, detector.radius, color='green', fill=False, linestyle='--', alpha=0.25)
        ax.add_artist(detector_circle)
    for self_vector in true_df['vector']:
        self_circle = plt.Circle(self_vector, self_region, color='blue', fill=False, linestyle='--', alpha=0.25)
        ax.add_artist(self_circle)
    plt.title("True and Fake")
    #print(detector_positions)
    plt.scatter(detector_positions[:, 0], detector_positions[:, 1], color='green', label='Detectors', alpha=0.25)
    plt.legend()
    plt.show()

# Function to generate points for a sphere around a detector
def create_sphere(center, radius, resolution=20):
    # Parametric angles
    theta = np.linspace(0, 2 * np.pi, resolution)
    phi = np.linspace(0, np.pi, resolution)
    theta, phi = np.meshgrid(theta, phi)
    
    # Spherical coordinates
    x = center[0] + radius * np.sin(phi) * np.cos(theta)
    y = center[1] + radius * np.sin(phi) * np.sin(theta)
    z = center[2] + radius * np.cos(phi)
    
    return x, y, z

def visualize_3d(true_df, fake_df, detector_set):
    detector_positions = np.array([detector.vector for detector in detector_set.detectors])
    true_cluster = np.array(true_df['vector'].tolist())
    fake_cluster = np.array(fake_df['vector'].tolist())

    # 3D Visualization
    detector_scatter = go.Scatter3d(
        x=detector_positions[:, 0],
        y=detector_positions[:, 1],
        z=detector_positions[:, 2],
        mode='markers',
        marker=dict(size=5, color='green'),
        name='Detectors'
    )

    true_scatter = go.Scatter3d(
        x=true_cluster[:, 0],
        y=true_cluster[:, 1],
        z=true_cluster[:, 2],
        mode='markers',
        marker=dict(size=5, color='blue'),
        name='True'
    )

    fake_scatter = go.Scatter3d(
        x=fake_cluster[:, 0],
        y=fake_cluster[:, 1],
        z=fake_cluster[:, 2],
        mode='markers',
        marker=dict(size=5, color='red'),
        name='Fake'
    )

    # Create spheres around detectors
    spheres = []
    for detector in detector_set.detectors:
        x, y, z = create_sphere(detector.vector, detector.radius)
        spheres.append(go.Mesh3d(
            x=x.flatten(),
            y=y.flatten(),
            z=z.flatten(),
            opacity=0.3,  # Transparency
            color='green',
            alphahull=0,
            name='Detector Sphere'
        ))

    # Create the figure and add the scatter plots and spheres
    fig = go.Figure(data=[detector_scatter, true_scatter, fake_scatter] + spheres)
    #fig = go.Figure(data=[true_scatter, fake_scatter])

    # Show the plot
    fig.show()


def euclidean_distance(a, b, a_radius, b_radius):
    #return np.sum(np.abs(a - b)) - a_radius - b_radius
    return np.linalg.norm(a - b) - a_radius - b_radius

def euclidian_distance_sparse_detector(detector, vector):
    sparse_vector = [vector[i] for i in detector.indices]
    return np.linalg.norm(sparse_vector - detector.vector) - detector.radius

def fast_cosine_similarity(a, b):
    # Dot product of vectors 'a' and 'b'
    dot_product = np.dot(a, b)
    
    # Norms of 'a' and 'b'
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    # Cosine similarity calculation
    return dot_product / (norm_a * norm_b)
    
    #return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def fast_cosine_distance(a, b):
    # Calculate cosine similarity first
    cosine_similarity = fast_cosine_similarity(a, b)
    
    # Convert cosine similarity to cosine distance
    return 1 - cosine_similarity

def fast_cosine_distance_with_radius(a, b, a_radius, b_radius):
    # Calculate cosine distance
    cosine_distance = fast_cosine_distance(a, b)
    
    # Subtract the radius from the distance
    adjusted_distance = cosine_distance - a_radius - b_radius
    #if adjusted_distance < 0:
    #    print('cosine distance', cosine_distance, a_radius, b_radius, adjusted_distance, b[0], b[1], a[0], a[1])
    # Ensure the distance is non-negative
    return adjusted_distance

def circle_overlap_area(center1, radius1, center2, radius2):
    # Calculate the distance between the centers
    distance = math.dist(center1, center2)
    
    # If the circles do not overlap
    if distance >= radius1 + radius2:
        return 0.0
    
    # If one circle is completely inside the other
    if distance <= abs(radius1 - radius2):
        return math.pi * min(radius1, radius2)**2
    
    # Calculate the overlap area using the circle segment formula
    part1 = radius1**2 * math.acos((distance**2 + radius1**2 - radius2**2) / (2 * distance * radius1))
    part2 = radius2**2 * math.acos((distance**2 + radius2**2 - radius1**2) / (2 * distance * radius2))
    part3 = 0.5 * math.sqrt((-distance + radius1 + radius2) * (distance + radius1 - radius2) * (distance - radius1 + radius2) * (distance + radius1 + radius2))
    
    return part1 + part2 - part3

# METRICS

def precision(tp, fp):
    if tp + fp == 0:
        return 0
    return tp / (tp + fp)

def recall(tp, fn):
    if tp + fn == 0:
        return 0
    return tp / (tp + fn)