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

# ---------------------- B-Spline Basis Functions ----------------------
class BsplineBasis1D:
    def __init__(self, center, half_extent):
        self.center = center
        self.half_extent = half_extent

    def evaluate(self, x):
        # Quadratic B-spline shape (compactly supported)
        return np.maximum(0, 1 - ((x - self.center) / self.half_extent)**2)

    def gradient(self, x):
        # A simple gradient: derivative of (1 - (x-center)^2/half_extent^2)
        diff = (x - self.center) / self.half_extent
        if np.abs(diff) < 1:
            return -2 * diff / self.half_extent
        return 0

class BsplineBasisND:
    def __init__(self, center, size, dimension):
        self.center = np.array(center)
        self.size = size
        self.half_extent = size / 2
        self.dimension = dimension
        self.basis_components = [BsplineBasis1D(self.center[i], self.half_extent) for i in range(dimension)]

    def evaluate(self, *args):
        val = 1.0
        for i in range(self.dimension):
            val *= self.basis_components[i].evaluate(args[i])
        return val

    def gradient(self, *args):
        grad = np.zeros(self.dimension)
        # For each coordinate, differentiate only that 1D basis while keeping the others fixed.
        for i in range(self.dimension):
            comp_grad = self.basis_components[i].gradient(args[i])
            prod = 1.0
            for j in range(self.dimension):
                if i != j:
                    prod *= self.basis_components[j].evaluate(args[j])
            grad[i] = comp_grad * prod
        return grad

# ---------------------- Multilevel System Assembly ----------------------
def collect_leaf_nodes_by_depth(octree):
    """
    Collect leaf nodes (nodes without children) grouped by their depth.
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

def compute_A_entry(basis_i, basis_j, node_i, node_j, finer_depth, alpha):
    """
    Compute an entry for the system matrix block.
    Uses an inner product of gradients plus a screening term.
    The scaling factor uses 2^(finer_depth)*alpha.
    """
    scaling_factor = (2 ** finer_depth) * alpha
    grad_i_at_j = basis_i.gradient(*node_j.center)
    grad_j_at_i = basis_j.gradient(*node_i.center)
    value = np.dot(grad_i_at_j, grad_j_at_i) + scaling_factor * (
        basis_i.evaluate(*node_j.center) * basis_j.evaluate(*node_i.center)
    )
    return value

def compute_b_entry(basis, node, points, normals, alpha):
    """
    Compute the right-hand side entry for a node.
    Sum contributions from all points (using the normal and gradient of the basis).
    """
    scaling_factor = (2 ** node.depth) * alpha
    value = 0.0
    for point, normal in zip(points, normals):
        grad_basis = basis.gradient(*point)
        value += scaling_factor * np.dot(normal, grad_basis)
    return value

def build_multilevel_system(octree, alpha, dimension, points, normals):
    """
    Build the per-depth system matrices A^d and right-hand sides b^d as well as
    the off-diagonal coupling matrices A^(d,d') for d' < d.
    Returns:
      nodes_by_depth: dict mapping depth d to list of nodes.
      A_blocks: dict with keys (d, d') (including d == d' for diagonal blocks).
      b_levels: dict mapping depth d to right-hand side vector.
    """
    nodes_by_depth = collect_leaf_nodes_by_depth(octree)
    depths = sorted(nodes_by_depth.keys())

    A_blocks = {}  # keys: (d, d') where d >= d'
    b_levels = {}

    # Build diagonal blocks and b^d for each depth.
    for d in depths:
        nodes_d = nodes_by_depth[d]
        Nd = len(nodes_d)
        A_diag = np.zeros((Nd, Nd))
        b_d = np.zeros(Nd)
        for i, node_i in enumerate(nodes_d):
            basis_i = BsplineBasisND(node_i.center, node_i.size, dimension)
            b_d[i] = compute_b_entry(basis_i, node_i, points, normals, alpha)
            for j, node_j in enumerate(nodes_d):
                basis_j = BsplineBasisND(node_j.center, node_j.size, dimension)
                A_diag[i, j] = compute_A_entry(basis_i, basis_j, node_i, node_j, d, alpha)
        A_blocks[(d, d)] = A_diag
        b_levels[d] = b_d

    # Build off-diagonal blocks for coupling (d > d').
    for d in depths:
        for d_prime in depths:
            if d > d_prime:
                nodes_d = nodes_by_depth[d]
                nodes_dp = nodes_by_depth[d_prime]
                Nd = len(nodes_d)
                Ndp = len(nodes_dp)
                A_off = np.zeros((Nd, Ndp))
                for i, node_i in enumerate(nodes_d):
                    basis_i = BsplineBasisND(node_i.center, node_i.size, dimension)
                    for j, node_j in enumerate(nodes_dp):
                        basis_j = BsplineBasisND(node_j.center, node_j.size, dimension)
                        A_off[i, j] = compute_A_entry(basis_i, basis_j, node_i, node_j, d, alpha)
                A_blocks[(d, d_prime)] = A_off

    return nodes_by_depth, A_blocks, b_levels

def solve_multilevel_system(A_blocks, b_levels, nodes_by_depth):
    """
    Solve the multilevel system using a cascadic strategy.
    For each depth d, adjust b^d by subtracting contributions from all coarser levels.
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

def evaluate_implicit_function(octree, x_levels, nodes_by_depth, eval_points, dimension):
    """
    Evaluate the implicit function
         χ(p) = sum_d sum_i x_i^d * B_i^d(p)
    at a collection of evaluation points.
    """
    values = np.zeros(len(eval_points))
    for d, nodes in nodes_by_depth.items():
        x_d = x_levels[d]
        for i, node in enumerate(nodes):
            basis = BsplineBasisND(node.center, node.size, dimension)
            for j, point in enumerate(eval_points):
                values[j] += x_d[i] * basis.evaluate(*point)
    return values

# ---------------------- Plotting for 2D ----------------------
def plot_implicit_function_2D(octree, x_levels, nodes_by_depth, points, dimension,normals = None, eval_resolution=100):
    # Create evaluation grid.
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, eval_resolution),
                         np.linspace(y_min, y_max, eval_resolution))
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])

    # Evaluate implicit function.
    implicit_values = evaluate_implicit_function(octree, x_levels, nodes_by_depth, grid_points, dimension)
    implicit_grid = implicit_values.reshape(xx.shape)

    # Plot both the implicit function and the point cloud with octree.
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Left: Implicit function contour.
    contour = ax[0].contourf(xx, yy, implicit_grid, cmap='viridis')
    fig.colorbar(contour, ax=ax[0], label='Implicit Function Value')
    ax[0].set_title('Implicit Function')
    ax[0].set_xlabel('X')
    ax[0].set_ylabel('Y')

    # Right: Points and octree structure (using provided plot_octree).
    ax[1].scatter(points[:, 0], points[:, 1], color='red', s=5, label='Points')
    if normals is not None:
        ax[1].quiver(points[:, 0], points[:, 1], normals[:, 0], normals[:, 1], color='blue', scale=20, label='Normals')
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
     # Create a 2D random polygon
    t = np.linspace(0, 2 * np.pi, 300)
    a, b = 10, 5  # Semi-major and semi-minor axes
    x = a * np.cos(t)
    y = b * np.sin(t)
    points = np.column_stack((x, y))

    # x = 16*np.sin(t)**3
    # y = 13*np.cos(t) - 5*np.cos(2*t) - 2*np.cos(3*t) - np.cos(4*t)
    # points = np.column_stack((x, y))


    # For this example, we define normals as outward-pointing from the centroid.
    normals = compute_local_PCA(points, points, radius=2)[1][:, :, 0]
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

    bbox_min = np.min(points, axis=0)
    bbox_max = np.max(points, axis=0)
    max_depth = 7          # maximum tree depth
    density_threshold = 0.03
    alpha = 4            # screening parameter

    # Build adaptive octree.
    octree = Octree(bbox_min, bbox_max, max_depth, points, density_threshold, dimension)
    octree.adapt()

    # Build the multilevel system.
    nodes_by_depth, A_blocks, b_levels = build_multilevel_system(octree, alpha, dimension, points, normals)

    # Solve using the cascadic strategy.
    x_levels = solve_multilevel_system(A_blocks, b_levels, nodes_by_depth)

    # Plot the implicit function and the octree structure.
    plot_implicit_function_2D(octree, x_levels, nodes_by_depth, points, dimension, normals)

if __name__ == '__main__':
    main()
