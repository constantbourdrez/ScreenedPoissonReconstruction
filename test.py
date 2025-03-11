import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from normals import compute_local_PCA  # Ensure this is defined in your environment
from ply import read_ply
from scipy.spatial import ConvexHull
# ---------------------- Adaptive Octree with Density Refinement ----------------------
class OctreeNode:
    def __init__(self, center, size, depth):
        self.center = np.array(center)
        self.size = size
        self.depth = depth
        self.children = None
        self.point_count = 0

    def subdivide(self):
        if self.children is None:
            half_size = self.size / 2
            # Offsets for the centers of the four subcells (2D)
            offsets = [(-0.5, -0.5), (-0.5, 0.5), (0.5, -0.5), (0.5, 0.5)]
            self.children = [
                OctreeNode(self.center + np.array(offset) * half_size, half_size, self.depth + 1)
                for offset in offsets
            ]

class Octree:
    def __init__(self, bbox_min, bbox_max, max_depth, points, density_threshold):
        self.points = points
        self.max_depth = max_depth
        self.density_threshold = density_threshold
        # Use the maximum side length as the initial cell size
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
            area = node.size**2
            density = node.point_count / area if area > 0 else 0
            # Subdivide if density is high and we haven't reached max depth
            if node.depth < self.max_depth and density > self.density_threshold:
                node.subdivide()
                nodes_to_check.extend(node.children)

# ---------------------- Fast Vectorized Quadratic Basis Functions ----------------------
class BsplineBasis1D:
    def __init__(self, center, half_extent):
        self.center = center
        self.half_extent = half_extent

    def evaluate(self, x):
        return np.maximum(0, 1 - ((x - self.center) / self.half_extent)**2)

class BsplineBasis2D:
    def __init__(self, center, size):
        self.center = np.array(center)
        self.size = size
        self.half_extent = size / 2
        self.basis_x = BsplineBasis1D(self.center[0], self.half_extent)
        self.basis_y = BsplineBasis1D(self.center[1], self.half_extent)

    def evaluate(self, X, Y):
        # X, Y are arrays (e.g. from meshgrid)
        return self.basis_x.evaluate(X) * self.basis_y.evaluate(Y)

# ---------------------- Utility: Plotting the Octree ----------------------
def plot_octree(node, ax):
    if node.children is None:
        lower = node.center - node.size / 2
        rect = plt.Rectangle(lower, node.size, node.size, fill=False, edgecolor='black')
        ax.add_patch(rect)
    else:
        for child in node.children:
            plot_octree(child, ax)

# ---------------------- Main Script ----------------------
# Create a 2D random polygon
t = np.linspace(0, 2 * np.pi, 100)
# a, b = 10, 5  # Semi-major and semi-minor axes
# x = a * np.cos(t)
# y = b * np.sin(t)
# points = np.column_stack((x, y))

x = 16*np.sin(t)**3
y = 13*np.cos(t) - 5*np.cos(2*t) - 2*np.cos(3*t) - np.cos(4*t)
points = np.column_stack((x, y))


normals = compute_local_PCA(points, points, radius=2)[1][:, :, 0]

# Build the adaptive octree based on point density
bbox_min = np.min(points, axis=0)
bbox_max = np.max(points, axis=0)
density_threshold = 0.02  # Adjust this threshold as needed
max_depth = 6
octree = Octree(bbox_min, bbox_max, max_depth, points, density_threshold)
octree.adapt()

# Gather all leaf nodes (cells that were not subdivided)
leaf_nodes = []
nodes = [octree.root]
while nodes:
    node = nodes.pop()
    if node.children is None:
        leaf_nodes.append(node)
    else:
        nodes.extend(node.children)

# Create a B-spline basis (here our parabolic hat) for each leaf cell
basis_functions = [BsplineBasis2D(node.center, node.size) for node in leaf_nodes]

# Create a grid over the domain and evaluate the summed basis functions (vectorized)
grid_x = np.linspace(bbox_min[0], bbox_max[0], 300)
grid_y = np.linspace(bbox_min[1], bbox_max[1], 300)
X, Y = np.meshgrid(grid_x, grid_y)
Z = np.zeros_like(X)
for basis in basis_functions:
    Z += basis.evaluate(X, Y)

# Plot the octree structure and the reconstructed function
fig, ax = plt.subplots(figsize=(8, 8))
plot_octree(octree.root, ax)
ax.scatter(points[:, 0], points[:, 1], color='red', s=5, zorder=10)
contour = ax.contourf(X, Y, Z, levels=50, alpha=0.6, cmap='viridis')
ax.quiver(points[:, 0], points[:, 1], 10*normals[:, 0], 10*normals[:, 1],
          color='red', angles='xy', scale_units='xy', scale=10, width=0.002, label="Normals")
plt.colorbar(contour, ax=ax)
ax.set_aspect('equal')
plt.title('Adaptive Octree with Vectorized B-spline Basis Functions')
plt.show()
