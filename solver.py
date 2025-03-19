import numpy as np
from b_spline import vectorized_bspline_eval, vectorized_bspline_gradient
from octree import collect_nodes_by_depth


# ---------------------- Multilevel System Assembly (Vectorized) ----------------------
def build_multilevel_system(octree, alpha, dimension, points, normals):
    """
    Assemble the system matrices A and right-hand sides b per level using vectorized operations.
    The screened term is added with scaling factor (2^d * alpha).
    """
    nodes_by_depth = collect_nodes_by_depth(octree)
    depths = sorted(nodes_by_depth.keys())
    #print('Depths:', depths)
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
        #print('Solving depth:', d)
        b_adjusted = b_levels[d].copy()
        for d_prime in depths:
            if d_prime < d and (d, d_prime) in A_blocks:
                b_adjusted -= A_blocks[(d, d_prime)] @ x_levels[d_prime]
        A_diag = A_blocks[(d, d)]
        x_levels[d] = np.linalg.solve(A_diag, b_adjusted)
    return x_levels
