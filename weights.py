import numpy as np
from b_spline import vectorized_bspline_eval
def InfluenceAtDepth(D, s, q, centers, half_extents):
    """
    Computes the influence of point s on point q at scale 2^-D (depth D).

    Parameters:
    - D: Depth level
    - s: Source point (influence source)
    - q: Query point (target of influence)
    - centers: Array of B-spline centers at depth D
    - half_extents: Half-extents for each center

    Returns:
    - Influence value at q due to s
    """
    if np.linalg.norm(s - q, np.inf) * (2**D) >= 3:  # q is outside influence region
        return 0
    else:
        s_sc = s * (2**D) + 0.5
        foo = s_sc - np.floor(s_sc)
        x, y = foo[0], foo[1]

        o1 = np.floor(s_sc) / (2**D)
        o4 = o1 + 1 / (2**D)
        o2 = np.array([o1[0], o4[1]])
        o3 = np.array([o4[0], o1[1]])

        o1 -= 0.5 / (2**D)
        o2 -= 0.5 / (2**D)
        o3 -= 0.5 / (2**D)
        o4 -= 0.5 / (2**D)

        def F_aux(o, t):
            """Computes B-spline value at q for offset o."""
            return np.sum(vectorized_bspline_eval(centers, half_extents, np.array([t])))

        return ((1 - x) * (1 - y) * F_aux(o1, q) +
                x * (1 - y) * F_aux(o2, q) +
                (1 - x) * y * F_aux(o3, q) +
                x * y * F_aux(o4, q))

def compute_influence(P, octree, nodes_by_depth, depths, max_depth, centers, half_extents, D_tilde):
    """
    Computes the influence of all points on P(i) for all i.

    Parameters:
    - P: (N, d) array of points.
    - octree: Spatial data structure (not used explicitly in this naive approach).
    - nodes_by_depth: Dictionary mapping depths to sets of octree nodes.
    - depths: Sorted list of available depths.
    - max_depth: Maximum depth in the octree.
    - centers: Centers of B-spline basis functions.
    - half_extents: Half-extents for each center.

    Returns:
    - W_all: Dictionary mapping depth -> array of influence values for each point.
    """
    N = P.shape[0]
    W_all = np.zeros((max_depth + 1, N))  # Influence values for all points at all depths
    for D in depths:
        print(f"Processing depth {D}...")

        for i in range(N):
            w_i = 0
            for j in range(N):  # Influence of all points on P[i]
                w_i += InfluenceAtDepth(D_tilde, P[j, :], P[i, :], centers, half_extents)

            W_all[D, i] = w_i  # Store influence for P[i] at depth D
    W_all = np.sum(W_all, axis=0)  

    return W_all
