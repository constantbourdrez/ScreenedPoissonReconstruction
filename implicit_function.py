import numpy as np
import matplotlib.pyplot as plt
from solver import build_multilevel_system, solve_multilevel_system
from octree import Octree
from b_spline import vectorized_bspline_eval
from normals import compute_local_PCA
from utils import plot_octree

# ---------------------- Implicit Function Evaluation (Vectorized) ----------------------
def evaluate_implicit_function(octree, x_levels, nodes_by_depth, eval_points, points, dimension):
    """
    Evaluate the implicit function χ(p) = sum_d sum_i x_i^d B_i^d(p)
    at many evaluation points.
    """
    values = np.zeros(eval_points.shape[0])
    control = np.zeros(points.shape[0])
    for d, nodes in nodes_by_depth.items():
        #print('depth:', d)
        centers = np.array([node.center for node in nodes])
        sizes = np.array([node.size for node in nodes])
        half_extents = sizes / 2.0
        E = vectorized_bspline_eval(centers, half_extents, eval_points)  # (N, P)
        #print('B-spline shape:', E.shape)
        B = vectorized_bspline_eval(centers, half_extents, points)  # (N, P)
        values += np.sum(x_levels[d][:, None] * E, axis=0)
        control += np.sum(x_levels[d][:, None] * B, axis=0)

    offset = np.mean(control)
    return values - offset

# ---------------------- Plotting for 2D ----------------------
def plot_implicit_function_2D(octree, x_levels, nodes_by_depth, points, dimension, normals=None, eval_resolution=100):
    # Create evaluation grid.
    x_min, y_min = 1.5 * np.min(points, axis=0)
    x_max, y_max = 1.5 * np.max(points, axis=0)
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, eval_resolution),
                         np.linspace(y_min, y_max, eval_resolution))
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])

    # Evaluate the implicit function.
    implicit_values = evaluate_implicit_function(octree, x_levels, nodes_by_depth, grid_points, points, dimension)
    implicit_grid = implicit_values.reshape(xx.shape)

    # Plot the implicit function (left) and the point cloud with the octree structure (right).
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Left: Contour plot of the implicit function.
    contour = ax[0].contourf(xx, yy, implicit_grid, cmap='viridis', extent=(-10., 10., -10., 10.))
    fig.colorbar(contour, ax=ax[0], label='Implicit Function Value')
    ax[0].contour(xx, yy, implicit_grid, levels=[0], colors='red', linestyles='dashed', label='Zero Level Set')
    ax[0].set_title('Implicit Function')
    ax[0].set_xlabel('X')
    ax[0].set_ylabel('Y')
    ax[0].xlim = (-10., 10.)
    ax[0].ylim = (-10., 10.)
    ax[0].set_aspect('equal')
    #ax[0].legend()

    # Right: Points (and normals) with the octree overlay.
    ax[1].scatter(points[:, 0], points[:, 1], color='red', s=5, label='Points')
    if normals is not None:
        ax[1].quiver(points[:, 0], points[:, 1], normals[:, 0], normals[:, 1],
                      color='blue', scale=20, label='Normals')
    plot_octree(octree.root, ax[1])
    ax[1].set_title('Point Cloud and Octree Structure')
    ax[1].set_xlabel('X')
    ax[1].set_ylabel('Y')
    ax[1].legend()

    plt.tight_layout()
    plt.show()

# ---------------------- Main Function ----------------------
def main():
    # Example settings for 2D.
    dimension = 2
    # Create a 2D ellipse (polygon)
    t = np.linspace(0, 2 * np.pi, 500)
    a, b = 10, 5  # Semi-major and semi-minor axes
    x = a * np.cos(t)
    y = b * np.sin(t)
    #points = np.column_stack((x, y))

    #points = []
    #points.append(np.array([np.linspace(-1, 1, 100), np.ones(100)]).T)
    #points.append(np.array([np.ones(100), np.linspace(1, -1, 100)]).T)
    #points.append(np.array([np.linspace(1, -1, 100), -np.ones(100)]).T)
    #points.append(np.array([-np.ones(100), np.linspace(-1, 1, 100)]).T)
    #points = np.concatenate(points)

    #noise = 0.05 * np.random.randn(*points.shape)
    #points += noise

    def generate_flower_point_cloud(N=1000, petals=6, noise=0.02):
        """Generate a 2D flower-like point cloud."""
        
        theta = np.linspace(0, 2 * np.pi, N)  # Angles from 0 to 2π
        r = 1 + 0.3 * np.cos(petals * theta)  # Polar function for flower shape
        
        # Convert to Cartesian coordinates
        x = r * np.cos(theta) + np.random.normal(0, noise, N)
        y = r * np.sin(theta) + np.random.normal(0, noise, N)
        
        return x, y

    # Generate and plot the flower point cloud
    x, y = generate_flower_point_cloud(N=1000, petals=6, noise=0.01)
    points = np.column_stack((x, y))

    # Compute normals using PCA (assumed to return normals correctly)
    normals = compute_local_PCA(points, points, radius=2)[1][:, :, 0]
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

    bbox_min = 1.5*np.min(points, axis=0)
    bbox_max = 1.5*np.max(points, axis=0)
    max_depth = 7  # maximum tree depth
    density_threshold = 0.5
    alpha = 16.          # screening parameter

    # Build the adaptive octree.
    octree = Octree(bbox_min, bbox_max, max_depth, points, density_threshold, dimension)
    octree.adapt()

    # Build the multilevel system using vectorized operations.
    nodes_by_depth, A_blocks, b_levels = build_multilevel_system(octree, alpha, dimension, points, normals)

    # Solve the system using a cascadic strategy.
    x_levels = solve_multilevel_system(A_blocks, b_levels, nodes_by_depth)

    # Plot the implicit function and the octree structure.
    plot_implicit_function_2D(octree, x_levels, nodes_by_depth, points, dimension, normals)

if __name__ == '__main__':
    main()
