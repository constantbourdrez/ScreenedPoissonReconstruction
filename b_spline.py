import numpy as np

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
    diff_scaled = diff / (3. * half_extents_reshaped)
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
    diff_scaled = diff / (3. * half_extents_reshaped)  # (N, P, d)
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
                              -2 * diff_scaled[:, :, k] / (3. * half_extents[:, None]),
                              0)
        # Compute product over all dimensions except k.
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.true_divide(prod_all, f_k)
            ratio[~np.isfinite(ratio)] = 0  # replace NaNs and infs with 0.
        prod_except = ratio
        grad[:, :, k] = factor * prod_except
    return grad
