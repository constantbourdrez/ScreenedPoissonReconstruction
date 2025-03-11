
# Import numpy package and name it "np"
import numpy as np

# Import library to plot in python
from matplotlib import pyplot as plt

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from ply import write_ply, read_ply

# Import time package
import time

import argparse
from typing import Optional, Tuple

# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#



def PCA(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the eigenvalues and eigenvectors of the covariance matrix of a point cloud.
    """
    barycenter = points.mean(axis=0)
    centered_points = points - barycenter
    cov_matrix = centered_points.T @ centered_points / points.shape[0]

    return np.linalg.eigh(cov_matrix)


def compute_local_PCA(
    query_points: np.ndarray,
    cloud_points: np.ndarray,
    nghbrd_search: str = "spherical",
    radius: Optional[float] = None,
    k: Optional[int] = None,
    d: int = 2,
    reference_direction: Optional[np.ndarray] = np.array([1, 0]),  # x-axis as a reference
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes PCA on the neighborhoods of all query_points in cloud_points.

    Returns:
        all_eigenvalues: (N, 2)-array of the eigenvalues associated with each query point.
        all_eigenvectors: (N, 2, 2)-array of the eigenvectors associated with each query point.
    """

    kdtree = KDTree(cloud_points)
    neighborhoods = (
        kdtree.query_radius(query_points, radius)
        if nghbrd_search.lower() == "spherical"
        else kdtree.query(query_points, k=k, return_distance=False)
        if nghbrd_search.lower() == "knn"
        else None
    )

    # checking the sizes of the neighborhoods and plotting the histogram
    if nghbrd_search.lower() == "spherical":
        neighborhood_sizes = [neighborhood.shape[0] for neighborhood in neighborhoods]
        print(
            f"Average size of neighborhoods: {np.mean(neighborhood_sizes):.4f}\n"
            f"Standard deviation: {np.std(neighborhood_sizes):.4f}\n"
            f"Min: {np.min(neighborhood_sizes)}, max: {np.max(neighborhood_sizes)}\n"
        )

    all_eigenvalues = np.zeros((query_points.shape[0], d))
    all_eigenvectors = np.zeros((query_points.shape[0], d, d))

    for i, point in enumerate(query_points):
        eigenvalues, eigenvectors = PCA(cloud_points[neighborhoods[i]])

        # In 2D, the eigenvector associated with the smallest eigenvalue is the normal.
        # Ensure the normal is oriented consistently with the reference direction.
        if np.dot(eigenvectors[:, 0], reference_direction) < 0:
            eigenvectors[:, 0] = -eigenvectors[:, 0]

        all_eigenvalues[i] = eigenvalues
        all_eigenvectors[i] = eigenvectors

    return all_eigenvalues, all_eigenvectors
