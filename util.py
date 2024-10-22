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

def visualize_3d(true_df, fake_df, detector_set, self_region):
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
    for self_sample in true_cluster:
        x, y, z = create_sphere(self_sample, self_region)
        spheres.append(go.Mesh3d(
            x=x.flatten(),
            y=y.flatten(),
            z=z.flatten(),
            opacity=0.3,  # Transparency
            color='blue',
            alphahull=0,
            name='Self Sphere'
        ))

    # Create the figure and add the scatter plots and spheres
    fig = go.Figure(data=[detector_scatter, true_scatter, fake_scatter] + spheres)
    #fig = go.Figure(data=[true_scatter, fake_scatter])

    # Show the plot
    fig.show()

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

def calculate_overlap(subject_vector_center, radius1, object_vector_center, radius2):
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


# M.O PLOTTING **********

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

def plot_schaffer(A, pareto_fronts):
    ''' Minimize f1(x) = x**2. Minimize f2(x) = (x - 2)**2 '''

    X = np.linspace(-A, A, 20*A)
    
    f1 = X**2
    f2 = (X - 2)**2

    # Creating subplots
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    # First subplot
    axs[0].plot(X, f1, label='$f1 = X^2$')
    axs[0].plot(X, f2, label='$f2 = (X - 2)^2$')
    axs[0].set_title('Plot of $f_1$ and $f_2$')
    axs[0].set_xlabel('$x$')
    axs[0].set_ylabel('$f_1, f_2$', rotation=0, labelpad=20)
    axs[0].legend()
    axs[0].grid(True)

    # Second subplot
    axs[1].plot(f2, f1, 'g-', label='$f1 = X^2$')  # f1 on primary y-axis
    axs[1].set_title('Objective space of $f_1$ and $f_2$')
    axs[1].set_xlabel('$f_2$')
    axs[1].set_ylabel('$f_1$', color='g', rotation=0, labelpad=20)
    axs[1].grid(True)

    # plot paretofronts
    max_x = -np.inf
    min_x = np.inf
    max_f1 = -np.inf
    max_f2 = -np.inf
    if pareto_fronts != None:
        for i, front in enumerate(pareto_fronts):
            f1 = []
            f2 = []
            cur_x = []
            for ind in front.individuals_sorted_by_distance:
                f1.append(ind.f1)
                f2.append(ind.f2)
                cur_x.append(ind.x)
            axs[0].scatter(np.concatenate((cur_x, cur_x)), np.concatenate((f1, f2)), label=f"{i+1}")
            axs[1].scatter(f2, f1, label=f'{i+1}')
            if np.max(cur_x) > max_x:
                max_x = np.max(cur_x)
            if np.min(cur_x) < min_x:
                min_x = np.min(cur_x)
            if np.max(f1) > max_f1:
                max_f1 = np.max(f1)
            if np.max(f2) > max_f2:
                max_f2 = np.max(f2)
    axs[1].legend(loc='upper right')
    
    # dynamic area to show
    axs[1].set_xlim(-1, max_f2)
    axs[1].set_ylim(-1, max_f1)
    axs[0].set_xlim(min_x - abs(0.25 * min_x), max_x + abs(0.25 * max_x))
    axs[0].set_ylim(-1, np.max([max_f1, max_f2]))
    # Displaying the plots
    plt.tight_layout()
    plt.show()



#plot_schaffer(None)

def animate_nsga_2_run(pareto_fronts_log, A=5):
    #animation_obj = animation.FuncAnimation(fig, animate, n_steps, fargs=(x_points, y_points), interval=1000) 

    def animate_fronts(i, pareto_fronts_log, f1, f2, X):
        axs[0].clear()
        axs[1].clear()
        axs[0].plot(X, f1, label='$f1 = X^2$')
        axs[0].plot(X, f2, label='$f2 = (X - 2)^2$')
        axs[1].plot(f2, f1, 'g-', label='$f1 = X^2$')  # f1 on primary y-axis

        # plot paretofronts
        max_x = -np.inf
        min_x = np.inf
        max_f1 = -np.inf
        max_f2 = -np.inf
        if pareto_fronts_log[i] != None:
            for j, front in enumerate(pareto_fronts_log[i]):
                cur_f1 = []
                cur_f2 = []
                cur_x = []
                for ind in front.individuals_sorted_by_distance:
                    cur_f1.append(ind.f1)
                    cur_f2.append(ind.f2)
                    cur_x.append(ind.x)

                axs[1].scatter(cur_f2, cur_f1, label=f'{j+1}')
                axs[0].scatter(np.concatenate((cur_x, cur_x)), np.concatenate((cur_f1, cur_f2)), label=f"{j+1}")

                if np.max(cur_x) > max_x:
                    max_x = np.max(cur_x)
                if np.min(cur_x) < min_x:
                    min_x = np.min(cur_x)
                if np.max(cur_f1) > max_f1:
                    max_f1 = np.max(cur_f1)
                if np.max(cur_f2) > max_f2:
                    max_f2 = np.max(cur_f2)

        axs[1].legend(loc='upper right')
        axs[0].legend(loc='upper right')

        # dynamic area to show
        axs[1].set_xlim(-1, max_f2)
        axs[1].set_ylim(-1, max_f1)
        axs[0].set_xlim(min_x - abs(0.25 * min_x), max_x + abs(0.25 * max_x))
        axs[0].set_ylim(-1, np.max([max_f1, max_f2]))

        text_top = np.max([max_f1, max_f2])
        axs[0].text(0, text_top + text_top/10, f"Generation {i+1} / {len(pareto_fronts_log)}")

    X = np.linspace(-A, A, 20*A)
    
    f1 = X**2
    f2 = (X - 2)**2

    # Creating subplots
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    # First subplot
    axs[0].plot(X, f1, label='$f1 = X^2$')
    axs[0].plot(X, f2, label='$f2 = (X - 2)^2$')
    axs[0].set_title('Plot of $f_1$ and $f_2$')
    axs[0].set_xlabel('$x$')
    axs[0].set_ylabel('$f_1, f_2$', rotation=0, labelpad=20)
    axs[0].legend()
    axs[0].grid(True)

    # Second subplot
    axs[1].plot(f2, f1, 'g-', label='$f1 = X^2$')  # f1 on primary y-axis
    axs[1].set_title('Objective space of $f_1$ and $f_2$')
    axs[1].set_xlabel('$f_2$')
    axs[1].set_ylabel('$f_1$', color='g', rotation=0, labelpad=20)
    axs[1].grid(True)

    # plot paretofronts
    max_x = -np.inf
    min_x = np.inf
    max_f1 = -np.inf
    max_f2 = -np.inf
    if pareto_fronts_log[0] != None:
        for i, front in enumerate(pareto_fronts_log[0]):
            cur_f1 = []
            cur_f2 = []
            cur_x = []
            for ind in front.individuals_sorted_by_distance:
                cur_f1.append(ind.f1)
                cur_f2.append(ind.f2)
                cur_x.append(ind.x)

            axs[1].scatter(cur_f2, cur_f1, label=f'{i+1}')
            axs[0].scatter(np.concatenate((cur_x, cur_x)), np.concatenate((cur_f1, cur_f2)), label=f"{i+1}")

            if np.max(cur_x) > max_x:
                max_x = np.max(cur_x)
            if np.min(cur_x) < min_x:
                min_x = np.min(cur_x)
            if np.max(cur_f1) > max_f1:
                max_f1 = np.max(cur_f1)
            if np.max(cur_f2) > max_f2:
                max_f2 = np.max(cur_f2)
            
    axs[1].legend(loc='upper right')
    axs[0].legend(loc='upper right')
    
    # dynamic area to show
    axs[1].set_xlim(-1, max_f2)
    axs[1].set_ylim(-1, max_f1)
    axs[0].set_xlim(min_x - abs(0.25 * min_x), max_x + abs(0.25 * max_x))
    axs[0].set_ylim(-1, np.max([max_f1, max_f2]))
    
    text_top = np.max([max_f1, max_f2])
    axs[0].text(0, text_top + text_top/10, f"Generation {1} / {len(pareto_fronts_log)}")

    animation_obj = animation.FuncAnimation(fig, animate_fronts, len(pareto_fronts_log), fargs=(pareto_fronts_log, f1, f2, X, ), interval=1500) 
    #writervideo = animation.FFMpegWriter(fps=1) 
    animation_obj.save("gifs/performance_animation.gif", dpi=300, writer=animation.PillowWriter(fps=1))
    #animation_obj.save('perfromance_animation.mp4', writer=writervideo)
    plt.close() 