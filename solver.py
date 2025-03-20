import numpy as np
from b_spline import vectorized_bspline_eval, vectorized_bspline_gradient
from octree import collect_nodes_by_depth

def build_multilevel_system(octree, alpha, dimension, points, V_field):
    """
    Assemble the system matrices A and right-hand side vectors b per level.

    Here, the accumulation field V_field (i.e. ~V) is used in place of point normals.
    For each node m at depth d, we want to compute:

        b_m = ⟨ ~V , ∇B_m ⟩ = sum_{n ∈ Interf(m)} ⟨ F_n ~V_n, ∇B_m ⟩,

    which, by separability, decomposes coordinate‐wise. In our implementation we
    approximate this by, for each node i at depth d:

        b_d[i] = sum_{p in points} [ V_field[d][i,0] * G_points[i, p, 0] +
                                    V_field[d][i,1] * G_points[i, p, 1] +
                                    V_field[d][i,2] * G_points[i, p, 2] ] * point_weight,

    where G_points[i, p, :] is the gradient of the B-spline for node i evaluated at point p.

    The assembled matrix A has diagonal blocks computed from the inner product
    ⟨∇B_n, ∇B_m⟩ plus a data-attachment term (scaled by (2^d * alpha)), and
    off-diagonal blocks are computed for interfering nodes (with d > d').

    Parameters:
      - octree: an octree (or quadtree) with nodes storing their centers and sizes.
      - alpha: a scalar data attachment parameter.
      - dimension: spatial dimension (e.g., 2 or 3).
      - points: (P, dimension) array of sample points.
      - V_field: a dictionary such that V_field[d] is an array of shape (Nd, dimension)
                 with the accumulation vectors for nodes at depth d.
      - area: a scalar used to compute integration weights.

    Returns:
      - nodes_by_depth: dictionary of nodes organized by depth.
      - A_blocks: dictionary with keys (d, d') representing blocks of the system matrix.
      - b_levels: dictionary with key d for the right-hand side vector at depth d.
    """
    nodes_by_depth = collect_nodes_by_depth(octree)
    depths = sorted(nodes_by_depth.keys())
    A_blocks = {}  # clés: (d, d') avec d >= d'
    b_levels = {}

    # Grille d'intégration pour le produit scalaire des gradients (pour A)
    step = 1.0 / 20
    volume_elements = step ** dimension
    grid_axes = [np.arange(0, 1, step) for _ in range(dimension)]
    grid = np.meshgrid(*grid_axes, indexing='ij')
    grid = np.stack(grid, axis=-1).reshape(-1, dimension)

    #point_weight = area / len(points)
    max_depth = max(depths)

    # Pré-calcul des centres et demi-extensions pour chaque niveau.
    centers_by_depth = {}
    half_extents_by_depth = {}
    for d in depths:
        nodes_d = nodes_by_depth[d]
        centers = np.array([node.center for node in nodes_d])
        sizes = np.array([node.size for node in nodes_d])
        centers_by_depth[d] = centers
        half_extents_by_depth[d] = sizes / 2.0

    # Pour chaque niveau, assembler le bloc diagonal de A et le vecteur b.
    for d in depths:
        nodes_d = nodes_by_depth[d]
        Nd = len(nodes_d)
        centers = centers_by_depth[d]
        half_extents = half_extents_by_depth[d]
        scaling =  (2**d)* alpha

        # Bloc diagonal A^(d,d)
        E = vectorized_bspline_eval(centers, half_extents, centers)  # (Nd, Nd)
        G = vectorized_bspline_gradient(centers, half_extents, centers)  # (Nd, Nd, dimension)
        grad_dot = np.sum(G * np.transpose(G, (1, 0, 2)), axis=2) # (Nd, Nd)
        A_diag = grad_dot + scaling * (E * E.T)
        A_blocks[(d, d)] = A_diag

        # Assemblage du vecteur b pour le niveau d:
        # G_points: gradients évalués aux points pour chaque nœud. Forme : (Nd, P, dimension)
        G_points = vectorized_bspline_gradient(centers, half_extents, points)
        b_d = np.zeros(Nd)
        for i in range(Nd):
            # V_field est de forme (P, 2) et G_points[i] est de forme (P, 2)
            # On calcule le produit scalaire point par point, puis on somme sur tous les points.
            prod = np.sum(V_field * G_points[i], axis=1)  # forme (P,)
            b_d[i] = np.sum(prod) * 2**(d+1)
        b_levels[d] = b_d

    # Assemblage des blocs hors-diagonaux pour d > d'
    for d in depths:
        for d_prime in depths:
            if d > d_prime:
                nodes_d = nodes_by_depth[d]
                nodes_dp = nodes_by_depth[d_prime]
                N1 = len(nodes_d)
                N2 = len(nodes_dp)
                centers1 = centers_by_depth[d]
                sizes1 = np.array([node.size for node in nodes_d])
                half_extents1 = sizes1 / 2.0
                centers2 = centers_by_depth[d_prime]
                sizes2 = np.array([node.size for node in nodes_dp])
                half_extents2 = sizes2 / 2.0
                scaling = (2 ** d) * alpha  # on utilise le niveau le plus fin

                E1 = vectorized_bspline_eval(centers1, half_extents1, centers2)  # (N1, N2)
                E2 = vectorized_bspline_eval(centers2, half_extents2, centers1).T  # (N1, N2)
                G1 = vectorized_bspline_gradient(centers1, half_extents1, centers2)  # (N1, N2, dimension)
                G2 = vectorized_bspline_gradient(centers2, half_extents2, centers1).transpose(1, 0, 2)  # (N1, N2, dimension)
                grad_dot = np.sum(G1 * G2, axis=2) # (N1, N2)
                A_off = grad_dot + scaling * (E1 * E2)
                A_blocks[(d, d_prime)] = A_off

    return nodes_by_depth, A_blocks, b_levels

def solve_multilevel_system(A_blocks, b_levels, nodes_by_depth):
    """
    Résout le système multilevel de manière cascadique.
    Pour chaque niveau d, ajuste b^d en soustrayant les contributions des niveaux coarses,
    puis résout A^(d,d) x^d = b^d.
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
