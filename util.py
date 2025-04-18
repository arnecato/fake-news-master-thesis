import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math
from scipy.spatial import Voronoi
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10
})

# Create 1D visualization with academic quality
def visualize_1d(true_df, fake_df, detector_set, self_region, algo, embedding):
    #detector_positions = np.array([detector.vector for detector in detector_set.detectors])
    true_cluster = np.array(true_df['vector'].tolist())
    fake_cluster = np.array(fake_df['vector'].tolist())

    fig, ax = plt.subplots(figsize=(8, 1.6), dpi=300)  # Adjusted size and high-res for print

    # True data intervals (self regions)
    for self_vector in true_cluster:
        ax.plot([self_vector[0] - self_region, self_vector[0] + self_region], 
                [0.4, 0.4], color='blue', linewidth=1, alpha=0.75, label="Real News" if 'Real News' not in ax.get_legend_handles_labels()[1] else "")

    # Fake data intervals
    for fake_vector in fake_cluster:
        ax.plot([fake_vector[0] - self_region, fake_vector[0] + self_region], 
                [0.2, 0.2], color='red', linewidth=1, alpha=0.75, label="Fake News" if 'Fake News' not in ax.get_legend_handles_labels()[1] else "")

    # Detector intervals
    for detector in detector_set.detectors:
        ax.plot([detector.vector[0] - detector.radius, detector.vector[0] + detector.radius], 
                [0, 0], color='green', linewidth=1, alpha=0.75, label="Detector" if 'Detector' not in ax.get_legend_handles_labels()[1] else "")

    # Aesthetic adjustments
    ax.set_yticks([0, 0.2, 0.4])
    ax.set_yticklabels(["Detectors", "Fake News", "Real News"])
    ax.set_ylim(-0.2, 0.6)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    # Title formatting
    algo = algo.upper()
    embedding = embedding.upper()
    ax.set_title(rf'{algo} {embedding} 1D Visualization', fontsize=14, fontweight='bold')

    # Ticks formatting
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(f'report/results/detector_plots/{algo}_{embedding}_1D_plot.png', dpi=300,
                bbox_inches='tight', format='png')
    #plt.show()

# Create 2D visualization
def visualize_2d_old(true_df, fake_df, detector_set, self_region):
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

def visualize_2d(true_df, fake_df, detector_set, self_region, algo, embedding):
    detector_positions = np.array([detector.vector for detector in detector_set.detectors])
    true_cluster = np.array(true_df['vector'].tolist())
    fake_cluster = np.array(fake_df['vector'].tolist())

    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)  # Adjusted size and high-res for print

    # True data points
    ax.scatter(true_cluster[:, 0], true_cluster[:, 1], 
               s=0.1, color='blue', alpha=0.25, label="Real News")

    # Fake data points
    ax.scatter(fake_cluster[:, 0], fake_cluster[:, 1], 
               s=0.1, color='red', alpha=0.25, label="Fake News")

    # Detectors
    ax.scatter(detector_positions[:, 0], detector_positions[:, 1],
               s=1, color='green', alpha=0.25, label='Detector')

    # Detector circles
    for detector in detector_set.detectors:
        detector_circle = plt.Circle(detector.vector, detector.radius,
                                     color='green', fill=False,
                                     linestyle='--', linewidth=0.5, alpha=0.25)
        ax.add_artist(detector_circle)

    # Self-region circles
    for self_vector in true_cluster:
        self_circle = plt.Circle(self_vector, self_region,
                                 color='blue', fill=True,
                                 linestyle='--', linewidth=0.5, alpha=0.75)
        ax.add_artist(self_circle)

    # Aesthetic adjustments
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', linewidth=0.5)
    
    # Title formatting
    algo = algo.upper()
    embedding = embedding.upper()
    ax.set_title(rf'{algo} {embedding} 2D', fontsize=14, fontweight='bold')
    
    # Ticks formatting
    import matplotlib.ticker as ticker
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    
    ax.legend(loc='upper left', fontsize=10)
    
    plt.savefig(f'report/results/detector_plots/{algo}_{embedding}_2D_plot.png', dpi=300,
                bbox_inches='tight', format='png')  # Save as vector format
    


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

def visualize_3d(true_df, fake_df, detector_set, self_region, algo, embedding_model, postfix):
    detector_positions = np.array([detector.vector for detector in detector_set.detectors]) if detector_set.detectors else np.array([])
    true_cluster = np.array(true_df['vector'].tolist())
    fake_cluster = np.array(fake_df['vector'].tolist())

    # 3D Visualization
    traces = []
    
    if detector_positions.size > 0:
        traces.append(go.Scatter3d(
            x=detector_positions[:, 0],
            y=detector_positions[:, 1],
            z=detector_positions[:, 2],
            mode='markers',
            marker=dict(size=3, color='green'),  # Smaller markers
            showlegend=False
        ))
    
    traces.append(go.Scatter3d(
        x=true_cluster[:, 0],
        y=true_cluster[:, 1],
        z=true_cluster[:, 2],
        mode='markers',
        marker=dict(size=3, color='blue'),  # Smaller markers
        showlegend=False
    ))
    
    traces.append(go.Scatter3d(
        x=fake_cluster[:, 0],
        y=fake_cluster[:, 1],
        z=fake_cluster[:, 2],
        mode='markers',
        marker=dict(size=3, color='red'),  # Smaller markers
        showlegend=False
    ))

    # Create spheres 
    spheres = []
    if detector_positions.size > 0:
        for detector in detector_set.detectors:
            x, y, z = create_sphere(detector.vector, detector.radius)
            spheres.append(go.Mesh3d(
                x=x.flatten(),
                y=y.flatten(),
                z=z.flatten(),
                opacity=0.3,  # Transparency
                color='green',
                alphahull=0,
                showlegend=False
            ))
    
    for self_sample in true_cluster:
        x, y, z = create_sphere(self_sample, self_region)
        spheres.append(go.Mesh3d(
            x=x.flatten(),
            y=y.flatten(),
            z=z.flatten(),
            opacity=0.3,  # Transparency
            color='blue',
            alphahull=0,
            showlegend=False
        ))

    # Create the figure and add the scatter plots and spheres
    fig = go.Figure(data=traces + spheres)
    
    # Set layout options for a compact appearance
    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            xaxis=dict(showgrid=True, showticklabels=True, zeroline=False),
            yaxis=dict(showgrid=True, showticklabels=True, zeroline=False),
            zaxis=dict(showgrid=True, showticklabels=True, zeroline=False),
            aspectmode="auto"  # Ensures optimal aspect ratio
        ),
        margin=dict(l=0, r=0, b=0, t=0),  # Remove extra space around the plot
        autosize=True,  # Ensures it scales properly in different views
    )
    
    # Save high-resolution images from different angles (top-down left, top-down right, bottom-up left, bottom-up right)
    angles = [(1.5, -1.5, 1.5), (0.1, 2.5, 0.1), (2.5, 0.1, 0.1), (-1.5, 1.5, 1.5)]
    for i, angle in enumerate(angles):
        fig.update_layout(
            scene_camera=dict(eye=dict(x=angle[0], y=angle[1], z=angle[2]))
        )
        if postfix is not None:
            postfix = f"_{postfix}"
        if len(detector_set.detectors) == 0:
            pio.write_image(fig, f"report/{algo}_{embedding_model}_3D_plot_angle_{i+1}{postfix}.png", width=1200, height=1000, scale=3)
        else:
            pio.write_image(fig, f"report/results/detector_plots/{algo}_{embedding_model}_3D_plot_angle_{i+1}{postfix}.png", width=1200, height=1000, scale=3)
    
    # Show the plot
    #fig.show()

def get_shared_feature_vectors(vector_a, vector_a_feature_index, vector_b, vector_b_feature_index):
    #print('vector_a_feature_index', vector_a_feature_index, 'vector_b_feature_index', vector_b_feature_index)
    self_vector = None
    detector_vector = None
    if vector_a is not None and vector_b is not None:
        if vector_a_feature_index is None:
            shared_features = vector_b_feature_index 
        elif vector_b_feature_index is None:
            shared_features = vector_a_feature_index
        else:
            shared_features = set(vector_a_feature_index).intersection(set(vector_b_feature_index))
        if len(shared_features) > 0:
            shared_features = list(shared_features)
            self_vector = np.zeros(len(shared_features))
            detector_vector = np.zeros(len(shared_features))
            for i, idx in enumerate(shared_features):
                #print(i, idx, vector_a, vector_b, vector_a_feature_index.index(idx), vector_b_feature_index.index(idx))
                self_vector[i] = vector_a[vector_a_feature_index.index(idx)]
                detector_vector[i] = vector_b[vector_b_feature_index.index(idx)]
    return self_vector, detector_vector, shared_features

def get_nearby_self(points, center, distance):
    # Calculate the minimum and maximum bounds for each dimension
    min_bounds = center - distance
    max_bounds = center + distance
    #print(points, center, distance, min_bounds, max_bounds)
    # Apply the conditions for all dimensions
    #print('Min bounds', min_bounds, 'Max bounds', max_bounds, 'Distance', distance)
    conditions = np.all((points >= min_bounds) & (points <= max_bounds), axis=1)
    potential_values = points[conditions]
    # Filter out points that are outside the distance (checking a square, while points can be diagonally further away)
    potential_values = potential_values[np.linalg.norm(potential_values - center, axis=1) < distance]
    return potential_values

def calculate_voronoi_points(detector_set):
    # calculate voronoi points
    if detector_set is not None and len(detector_set.detectors) > 5:
        detector_points = np.array([detector.vector for detector in detector_set.detectors])
        vor = Voronoi(detector_points)
        #vor.vertices = np.clip(vor.vertices, self.feature_low, self.feature_max)
        return vor.vertices
    else:
        return None
    
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

def fast_cosine_distance_with_radius(a, b, a_radius, b_radius, feature_index):
    # Calculate cosine distance
    cosine_distance = fast_cosine_distance(a, b)
    
    # Subtract the radius from the distance
    adjusted_distance = cosine_distance - a_radius - b_radius
    #if adjusted_distance < 0:
    #    print('cosine distance', cosine_distance, a_radius, b_radius, adjusted_distance, b[0], b[1], a[0], a[1])
    # Ensure the distance is non-negative
    return adjusted_distance

def calculate_self_region(self_points):
    self_distances = []
    for self_point_1 in self_points: #.itertuples(index=False, name=None):
        closest_distance = 999999.0
        for self_point_2 in self_points: #.itertuples(index=False, name=None):
            distance = euclidean_distance(self_point_1, self_point_2, 0, 0)
            if distance < closest_distance and distance != 0:
                closest_distance = distance
        #print(f'distance found {closest_distance}', row_1st[1], row_2nd[1])
        if distance != 0: # avoid adding distance to itself
            self_distances.append(closest_distance)
    return np.mean(self_distances) 

def hypersphere_volume(radius, dimension):
    if dimension == 1:
        return 2 * radius
    return (math.pi ** (dimension / 2) * radius ** dimension) / math.gamma((dimension / 2) + 1)

def hypersphere_overlap(r1, r2, distance, dimension):
    """
    Calculate the overlap volume between two n-dimensional hyperspheres.
    
    :param r1: float, radius of the first hypersphere
    :param r2: float, radius of the second hypersphere
    :param distance: float, distance between the centers of the hyperspheres
    :param dimension: int, dimension of the hyperspheres (2, 3, or 4)
    :return: float, overlap volume
    """
    if distance >= r1 + r2:
        return 0.0  # No overlap
    elif distance <= abs(r1 - r2):
        return hypersphere_volume(min(r1, r2), dimension)  # One is completely inside the other
    if dimension == 1:  # Overlap length for line segments
        return min(r1 + r2 - distance, min(r1, r2))
    if dimension == 2:  # Overlap area for circles
        part1 = r1**2 * math.acos((distance**2 + r1**2 - r2**2) / (2 * distance * r1))
        part2 = r2**2 * math.acos((distance**2 + r2**2 - r1**2) / (2 * distance * r2))
        part3 = 0.5 * math.sqrt((-distance + r1 + r2) * (distance + r1 - r2) * (distance - r1 + r2) * (distance + r1 + r2))
        return part1 + part2 - part3
    
    elif dimension == 3:  # Overlap volume for spheres
        return (math.pi * (r1 + r2 - distance)**2 / (12 * distance)) * (
            distance**2 + 2 * distance * (r1 + r2) - 3 * (r1**2 + r2**2) + 6 * r1 * r2
        )
    
    elif dimension == 4:  # Approximation for 4D hypersphere overlap
        # Using integral-based approximation for higher dimensions
        r1_vol = hypersphere_volume(r1, 4)
        r2_vol = hypersphere_volume(r2, 4)
        if distance <= abs(r1 - r2):  # One hypersphere is inside the other
            return min(r1_vol, r2_vol)
        combined_radius = max(0, (r1 + r2 - distance) / 2)
        return hypersphere_volume(combined_radius, 4)
    
    else:
        raise ValueError("Overlap formulas are only implemented for dimensions 2, 3, and 4.")

def total_detector_hypersphere_volume(detector_set):
    if len(detector_set.detectors) == 0:
        return 0
    volume = 0
    overlap = 0
    for i, detector_1 in enumerate(detector_set.detectors): #self.true_df.itertuples(index=False, name=None)):
        volume += hypersphere_volume(detector_1.radius, len(detector_1.vector))
        for j, detector_2 in enumerate(detector_set.detectors): #enumerate(self.true_df.itertuples(index=False, name=None)):
            if i != j:
                overlap += hypersphere_overlap(detector_1.radius, detector_2.radius, math.dist(detector_1.vector, detector_2.vector), len(detector_1.vector)) / 2.0 #alculate_overlap(row_1st[1], self.self_region, row_2nd[1], self.self_region) / 2.0 # only count overlap for one of them
    return volume - overlap

def calculate_radius_overlap(subject_vector_center, radius1, object_vector_center, radius2):
    # Calculate the distance between the centers
    distance = math.dist(subject_vector_center, object_vector_center)
    
    # If no overlap, nothing needs to be reduced
    if distance >= radius1 + radius2:
        return 0.0
    # complete coverage of subject by object
    elif distance <= radius2:
        #print('complete coverage', distance, radius2)
        return abs(radius1) # subject is completely covered by object
    # partial coverage of subject by object
    elif distance < radius1 + radius2:
        overlap_amount = (radius1 + radius2) - distance
        reduction_amount = overlap_amount / len(subject_vector_center) # adjust for dimensionality
        #print('distance', distance, 'radius1', radius1, 'radius2', radius2, reduction_amount)
        return reduction_amount
    
    print('ops', subject_vector_center, object_vector_center, radius1, radius2)

# METRICS

def precision(tp, fp):
    if tp + fp == 0:
        return 0
    return tp / (tp + fp)

def recall(tp, fn):
    if tp + fn == 0:
        return 0
    return tp / (tp + fn)

def f1_score(tp, fp, fn):
    p = precision(tp, fp)
    r = recall(tp, fn)
    if p + r == 0:
        return 0
    return 2 * (p * r) / (p + r)

v = hypersphere_volume(0.04318423289042272, 2)
o = hypersphere_overlap(1, 1, 1, 2)
print(v, o)