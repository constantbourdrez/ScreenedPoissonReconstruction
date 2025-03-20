import numpy as np
from b_spline import vectorized_bspline_eval
from octree import Octree, OctreeNode, collect_leaf_nodes_by_depth, collect_nodes_by_depth
from normals import compute_local_PCA
from weights import compute_influence
from utils import plot_octree
from sklearn.neighbors import NearestNeighbors


def estimate_surface_area(points, k=6):
    """
    Estimate the surface area of the reconstructed surface from a set of points,
    using a local sampling density heuristic inspired by Kazhdan et al. (2006).

    Parameters:
      points (np.ndarray): Array of shape (N, 3) containing the sample points.
      k (int): Number of nearest neighbors to use in the density estimation.

    Returns:
      float: An estimate of the surface area.
    """
    # Build a kd-tree for the points
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(points)
    distances, _ = nbrs.kneighbors(points)
    # The first neighbor is the point itself (distance 0), so use the k-th neighbor.
    kth_distances = distances[:, k]

    # Estimate local area at each point assuming a circular patch:
    #   local_area ≈ π * (kth_distance)^2
    local_areas = np.pi * kth_distances**2

    # Since these local areas overlap, average them appropriately.
    # A simple heuristic is to take the sum and divide by k.
    estimated_area = np.sum(local_areas) / k
    return estimated_area

def compute_diffusion_depths(density, max_depth=10):
    """
    Déduit la profondeur de diffusion Prof(p) à partir de la densité W_D(p).
    Plus la densité est faible, plus la diffusion se fait à une grande échelle.
    """
    return np.minimum(max_depth, max_depth + np.log(density / np.mean(density)) / np.log(4))

def diffuse_normals(points, normals, diffusion_depths, octree, nodes_by_depth, centers_by_depth, half_extents_by_depth, density):
    """
    Diffuse les normales à l'échelle 2^(-Prof(p)) en fonction du support de F.
    """
    accumulated_normals = np.zeros_like(normals)
    print()
    for i, point in enumerate(points):
        depth = diffusion_depths[i]
        if depth not in nodes_by_depth:
            continue

        centers = centers_by_depth[depth]
        half_extents = half_extents_by_depth[depth]

        # Poids d'influence du point sur les voisins
        weights = vectorized_bspline_eval(centers, half_extents, point[None, :])

        # Diffusion de la normale
        accumulated_normals =np.divide(np.sum(weights[:, None] * normals, axis=0), density[:, None])

    return accumulated_normals

def loadV(octree, alpha, dimension, points, normals, area):
    """
    Charge le vecteur d'accumulation selon l'octree.
    """
    nodes_by_depth = collect_nodes_by_depth(octree)
    depths = sorted(nodes_by_depth.keys())
    max_depth = max(depths)

    # Pré-calcul des centres et demi-extensions
    centers_by_depth = {d: np.array([node.center for node in nodes_by_depth[d]]) for d in depths}
    half_extents_by_depth = {d: np.array([node.size for node in nodes_by_depth[d]]) / 2.0 for d in depths}

    max_depth = max(depths)
    centers = np.array([node.center for node in nodes_by_depth[max_depth]])
    half_extents = np.array([node.size for node in nodes_by_depth[max_depth]]) / 2.0
    D_tilde = 3
    density = compute_influence(points, octree, nodes_by_depth, depths, max_depth, centers, half_extents, D_tilde)
    # Étape 2 : Calcul des profondeurs de diffusion Prof(p)
    diffusion_depths = compute_diffusion_depths(density,max_depth=max_depth)
    # Étape 3 : Diffusion des normales
    accumulated_normals = diffuse_normals(points, normals, diffusion_depths, octree, nodes_by_depth, centers_by_depth, half_extents_by_depth, density)
    return accumulated_normals


def main():
    # Example settings for 3D.
    dimension = 2
    t = np.linspace(0, 2 * np.pi, 500)
    a, b = 10, 5  # Semi-major and semi-minor axes
    x = a * np.cos(t)
    y = b * np.sin(t)
    points = np.column_stack((x, y))
    np.random.shuffle(points)
    points = points[:int(points.shape[0]/2), :]


    normals = compute_local_PCA(points, points, radius=2, d = dimension)[1][:, :, 0]
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    # estimate D_tilde such as the 2^-D is of the magnitude of the sampling length
    sampling_length = np.mean(np.array([np.linalg.norm(points[i] - points[i+1]) for i in range(points.shape[0]-1)]))
    D_tilde = int(-np.log2(sampling_length))
    # Define bounding box and octree parameters.
    bbox_min = np.min(points, axis=0) - 0.5
    bbox_max = np.max(points, axis=0) + 0.5
    max_depth = 6
    density_threshold = 0.2
    alpha = 4.
    #subsample the points but randomly, I want more dense zones
    # Build the adaptive octree.
    octree = Octree(bbox_min, bbox_max, max_depth, points, density_threshold, dimension)
    octree.adapt()
    print('Octree done')
    area = estimate_surface_area(points, k=6)
    a = loadV(octree, alpha, dimension, points, normals, area)
    #plot points, accumulated normals, normals and octree
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.scatter(points[:, 0], points[:, 1], c='blue', s=1)
    ax.quiver(points[:, 0], points[:, 1], normals[:, 0], normals[:, 1], color='red')
    ax.quiver(points[:, 0], points[:, 1], a[:, 0], a[:, 1], color='green')
    plot_octree(ax, octree.root)
    plt.show()
if __name__ == '__main__':
    main()
