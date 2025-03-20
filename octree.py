import numpy as np



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
    def contains(self, point):
        """
        Check if the given point is within the bounds of this node.
        """
        half_size = self.size / 2.0
        return all(abs(point[i] - self.center[i]) <= half_size for i in range(len(point)))

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
            if node.depth < self.max_depth and node.point_count > 1:
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

def collect_nodes_by_depth(octree):
    """
    Collect all nodes grouped by their depth.
    """
    nodes_by_depth = {}
    def collect(node):
        nodes_by_depth.setdefault(node.depth, []).append(node)
        if node.children is not None:
            for child in node.children:
                collect(child)
    collect(octree.root)
    return nodes_by_depth
