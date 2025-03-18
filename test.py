import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import cg
from normals import compute_local_PCA
from utils import plot_octree

# ---------------------- Adaptive Octree with Density Refinement ----------------------
class OctreeNode:
    def __init__(self, center, size, depth):
        self.center = np.array(center)
        self.size = size
        self.depth = depth
        self.children = None
        self.point_count = 0

    def subdivide(self, dimension):
        if self.children is None:
            half_size = self.size / 2
            if dimension == 2:
                offsets = [(-0.5, -0.5), (-0.5, 0.5), (0.5, -0.5), (0.5, 0.5)]
            elif dimension == 3:
                offsets = [(-0.5, -0.5, -0.5), (-0.5, -0.5, 0.5), (-0.5, 0.5, -0.5), (-0.5, 0.5, 0.5),
                           (0.5, -0.5, -0.5), (0.5, -0.5, 0.5), (0.5, 0.5, -0.5), (0.5, 0.5, 0.5)]
            self.children = [
                OctreeNode(self.center + np.array(offset) * half_size, half_size, self.depth + 1)
                for offset in offsets
            ]

class Octree:
    def __init__(self, bbox_min, bbox_max, max_depth, points, density_threshold, dimension):
        self.points = points
        self.max_depth = max_depth
        self.density_threshold = density_threshold
        self.dimension = dimension
        init_size = np.max(bbox_max - bbox_min)
        self.root = OctreeNode((bbox_min + bbox_max) / 2, init_size, 0)

    def adapt(self):
        nodes_to_check = [self.root]
        while nodes_to_check:
            node = nodes_to_check.pop()
            lower = node.center - node.size / 2
            upper = node.center + node.size / 2
            mask = np.all((self.points >= lower) & (self.points <= upper), axis=1)
            node.point_count = np.sum(mask)
            area = node.size**self.dimension
            density = node.point_count / area if area > 0 else 0
            if node.depth < self.max_depth and density > self.density_threshold:
                node.subdivide(self.dimension)
                nodes_to_check.extend(node.children)

def collect_leaf_nodes_by_depth(octree):
    """
    Recursively collect leaf nodes (nodes with no children) grouped by their depth.
    """
    nodes_by_depth = {}
    def collect(node):
        if node.children is None:
            nodes_by_depth.setdefault(node.depth, []).append(node)
        else:
            for child in node.children:
                collect(child)
    collect(octree.root)
    return nodes_by_depth

# ---------------------- Vectorized B-Spline Basis Functions ----------------------
def vectorized_bspline_eval(centers, half_extents, points):
    """
    Evaluate quadratic B-spline basis functions at many points.
    centers: (N, d) array for N nodes
    half_extents: (N,) array (each = size/2)
    points: (P, d) array of evaluation points
    Returns: (N, P) array where entry [i, p] = basis_i(points[p])
    """
    # Compute differences: shape (N, P, d)
    diff = points[None, :, :] - centers[:, None, :]
    # Normalize by half_extent (reshaped for broadcasting)
    half_extents_reshaped = half_extents[:, None, None]
    diff_scaled = diff / half_extents_reshaped
    # 1D quadratic B-spline: f(t) = max(0, 1 - t^2)
    vals = np.maximum(0, 1 - diff_scaled**2)  # shape (N, P, d)
    return np.prod(vals, axis=2)

def vectorized_bspline_gradient(centers, half_extents, points):
    """
    Compute the gradient of the quadratic B-spline basis functions in a vectorized way.
    centers: (N, d) array for N nodes.
    half_extents: (N,) array (each = size/2).
    points: (P, d) array of evaluation points.
    Returns: (N, P, d) array where [i, p, k] is the k-th component of the gradient
             of basis_i evaluated at points[p].
    """
    N, d = centers.shape
    P = points.shape[0]
    diff = points[None, :, :] - centers[:, None, :]  # (N, P, d)
    half_extents_reshaped = half_extents[:, None, None]  # (N, 1, 1)
    diff_scaled = diff / half_extents_reshaped  # (N, P, d)
    # Compute the 1D factors f(t) = max(0, 1 - t^2) for each dimension.
    f = np.maximum(0, 1 - diff_scaled**2)  # (N, P, d)
    prod_all = np.prod(f, axis=2)  # (N, P)
    grad = np.zeros((N, P, d))
    for k in range(d):
        f_k = f[:, :, k]  # (N, P)
        # Derivative of (1 - t^2) is -2*t (only valid for |t| < 1).
        # Reshape half_extents to (N, 1) to broadcast over P.
        with np.errstate(divide='ignore', invalid='ignore'):
            factor = np.where(np.abs(diff_scaled[:, :, k]) < 1,
                              -2 * diff_scaled[:, :, k] / half_extents[:, None],
                              0)
        # Compute product over all dimensions except k.
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.true_divide(prod_all, f_k)
            ratio[~np.isfinite(ratio)] = 0  # replace NaNs and infs with 0.
        prod_except = ratio
        grad[:, :, k] = factor * prod_except
    return grad


# ---------------------- Multilevel System Assembly (Vectorized) ----------------------
def build_multilevel_system(octree, alpha, dimension, points, normals):
    """
    Assemble the system matrices A and right-hand sides b per level using vectorized operations.
    The screened term is added with scaling factor (2^d * alpha).
    """
    nodes_by_depth = collect_leaf_nodes_by_depth(octree)
    depths = sorted(nodes_by_depth.keys())
    A_blocks = {}  # keys: (d, d') with d >= d'
    b_levels = {}

    # For each depth, assemble the diagonal block and right-hand side.
    for d in depths:
        nodes_d = nodes_by_depth[d]
        Nd = len(nodes_d)
        centers = np.array([node.center for node in nodes_d])
        sizes = np.array([node.size for node in nodes_d])
        half_extents = sizes / 2.0
        scaling = (2 ** d) * alpha

        # Diagonal block A^(d,d)
        E = vectorized_bspline_eval(centers, half_extents, centers)  # (Nd, Nd)
        G = vectorized_bspline_gradient(centers, half_extents, centers)  # (Nd, Nd, dimension)
        # For each pair (i, j), we need dot(G_i(center_j), G_j(center_i)).
        grad_dot = np.sum(G * np.transpose(G, (1, 0, 2)), axis=2)  # (Nd, Nd)
        A_diag = grad_dot + scaling * (E * E.T)
        A_blocks[(d, d)] = A_diag

        # Assemble b vector for this depth:
        G_points = vectorized_bspline_gradient(centers, half_extents, points)  # (Nd, P, dimension)
        # For each node i, dot the gradient at each point with the point's normal, then sum over all points.
        dot_prod = np.sum(G_points * normals[None, :, :], axis=2)  # (Nd, P)
        b_d = scaling * np.sum(dot_prod, axis=1)  # (Nd,)
        b_levels[d] = b_d

    # Off-diagonal blocks for d > d'
    for d in depths:
        for d_prime in depths:
            if d > d_prime:
                nodes_d = nodes_by_depth[d]
                nodes_dp = nodes_by_depth[d_prime]
                N1 = len(nodes_d)
                N2 = len(nodes_dp)
                centers1 = np.array([node.center for node in nodes_d])
                sizes1 = np.array([node.size for node in nodes_d])
                half_extents1 = sizes1 / 2.0
                centers2 = np.array([node.center for node in nodes_dp])
                sizes2 = np.array([node.size for node in nodes_dp])
                half_extents2 = sizes2 / 2.0
                scaling = (2 ** d) * alpha  # use the finer level's depth

                # Deeper nodes' basis functions evaluated at coarser nodes' centers.
                E1 = vectorized_bspline_eval(centers1, half_extents1, centers2)  # (N1, N2)
                # Coarser nodes' basis functions evaluated at deeper nodes' centers.
                E2 = vectorized_bspline_eval(centers2, half_extents2, centers1).T  # (N1, N2)
                # Gradients:
                G1 = vectorized_bspline_gradient(centers1, half_extents1, centers2)  # (N1, N2, dimension)
                G2 = vectorized_bspline_gradient(centers2, half_extents2, centers1).transpose(1, 0, 2)  # (N1, N2, dimension)
                grad_dot = np.sum(G1 * G2, axis=2)  # (N1, N2)
                A_off = grad_dot + scaling * (E1 * E2)
                A_blocks[(d, d_prime)] = A_off

    return nodes_by_depth, A_blocks, b_levels

def solve_multilevel_system(A_blocks, b_levels, nodes_by_depth):
    """
    Solve the multilevel system in a cascadic fashion.
    For each depth d, adjust b^d by subtracting contributions from coarser levels.
    """
    depths = sorted(nodes_by_depth.keys())
    x_levels = {}
    for d in depths:
        b_adjusted = b_levels[d].copy()
        for d_prime in depths:
            if d_prime < d and (d, d_prime) in A_blocks:
                b_adjusted -= A_blocks[(d, d_prime)] @ x_levels[d_prime]
        A_diag = A_blocks[(d, d)]
        x_levels[d] = np.linalg.solve(A_diag, b_adjusted)
    return x_levels

# ---------------------- Implicit Function Evaluation (Vectorized) ----------------------
def evaluate_implicit_function(octree, x_levels, nodes_by_depth, eval_points, dimension):
    """
    Evaluate the implicit function χ(p) = sum_d sum_i x_i^d B_i^d(p)
    at many evaluation points.
    """
    values = np.zeros(eval_points.shape[0])
    for d, nodes in nodes_by_depth.items():
        centers = np.array([node.center for node in nodes])
        sizes = np.array([node.size for node in nodes])
        half_extents = sizes / 2.0
        E = vectorized_bspline_eval(centers, half_extents, eval_points)  # (N, P)
        values += np.sum(x_levels[d][:, None] * E, axis=0)
    return values

# ---------------------- Plotting for 2D ----------------------
def plot_implicit_function_2D(octree, x_levels, nodes_by_depth, points, dimension, normals=None, eval_resolution=100):
    # Create evaluation grid.
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, eval_resolution),
                         np.linspace(y_min, y_max, eval_resolution))
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])

    # Evaluate the implicit function.
    implicit_values = evaluate_implicit_function(octree, x_levels, nodes_by_depth, grid_points, dimension)
    implicit_grid = implicit_values.reshape(xx.shape)

    # Plot the implicit function (left) and the point cloud with the octree structure (right).
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Left: Contour plot of the implicit function.
    contour = ax[0].contourf(xx, yy, implicit_grid, cmap='viridis')
    fig.colorbar(contour, ax=ax[0], label='Implicit Function Value')
    ax[0].set_title('Implicit Function')
    ax[0].set_xlabel('X')
    ax[0].set_ylabel('Y')

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
    t = np.linspace(0, 2 * np.pi, 300)
    a, b = 10, 5  # Semi-major and semi-minor axes
    x = a * np.cos(t)
    y = b * np.sin(t)
    points = np.column_stack((x, y))

    # Compute normals using PCA (assumed to return normals correctly)
    normals = compute_local_PCA(points, points, radius=2)[1][:, :, 0]
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

    bbox_min = np.min(points, axis=0)
    bbox_max = np.max(points, axis=0)
    max_depth = 7          # maximum tree depth
    density_threshold = 0.03
    alpha = 4              # screening parameter

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
