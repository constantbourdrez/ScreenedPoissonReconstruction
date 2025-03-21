import numpy as np

def vectorized_bspline_eval(centers, half_extents, points):
    """
    Evaluate quadratic B-spline basis functions at many points in 2D.
    centers: (N, 2) array for N nodes.
    half_extents: (N,) array (each = size/2).
    points: (P, 2) array of evaluation points.
    Returns: (N, P) array where entry [i, p] = basis_i(points[p])
    """
    # Compute differences: shape (N, P, 2)
    diff = points[None, :, :] - centers[:, None, :]
    # Normalize by half_extents (reshaped for broadcasting)
    diff_scaled = diff / (2.0 * half_extents[:, None, None])

    # Use absolute value since the function is even: f(x) = f(|x|)
    abs_diff = np.abs(diff_scaled)

    # Vectorized definition of f:
    # If abs_diff >= 1.5, value is 0.
    # If 0.5 <= abs_diff < 1.5, value is ((abs_diff - 1.5)**2) / 2.
    # Otherwise, value is -abs_diff**2 + 0.75.
    f_vals = np.where(
        abs_diff >= 1.5,
        0.0,
        np.where(
            abs_diff >= 0.5,
            ((abs_diff - 1.5)**2) / 2.0,
            -abs_diff**2 + 0.75
        )
    )

    # Multiply the two 1D factors to get the 2D B-spline basis.
    return np.prod(f_vals, axis=2)


def vectorized_bspline_gradient(centers, half_extents, points):
    """
    Compute the gradient of the quadratic B-spline basis functions in 2D.
    centers: (N, 2) array for N nodes.
    half_extents: (N,) array (each = size/2).
    points: (P, 2) array of evaluation points.
    Returns: (N, P, 2) array where [i, p, k] is the k-th component of the gradient
             of basis_i evaluated at points[p].
    """
    N, d = centers.shape  # assuming d == 2
    # Compute differences and scale them as before.
    diff = points[None, :, :] - centers[:, None, :]
    diff_scaled = diff / (2.0 * half_extents[:, None, None])

    # Define vectorized f (as above) for each dimension.
    abs_diff = np.abs(diff_scaled)
    f_vals = np.where(
        abs_diff >= 1.5,
        0.0,
        np.where(
            abs_diff >= 0.5,
            ((abs_diff - 1.5)**2) / 2.0,
            -abs_diff**2 + 0.75
        )
    )

    # Define vectorized grad_f.
    # For x, let s = sign(x) and y = |x|. Then:
    # if y >= 1.5: 0, if 0.5 <= y < 1.5: y - 1.5, else: -2*y.
    s = np.sign(diff_scaled)
    g_vals = np.where(
        abs_diff >= 1.5,
        0.0,
        np.where(
            abs_diff >= 0.5,
            abs_diff - 1.5,
            -2.0 * abs_diff
        )
    )
    grad_f_vals = s * g_vals  # this is grad_f(x) in a vectorized form

    # For 2D, the gradient for each node i and point p is given by:
    # [ grad_f(diff_scaled[i,p,0]) * f(diff_scaled[i,p,1]),
    #   f(diff_scaled[i,p,0]) * grad_f(diff_scaled[i,p,1]) ]
    # and we need to divide by (2 * half_extents[i]) for each node.
    grad = np.empty_like(diff_scaled)
    # Gradient with respect to x component
    grad[..., 0] = (grad_f_vals[..., 0] * f_vals[..., 1]) / (2.0 * half_extents[:, None])
    # Gradient with respect to y component
    grad[..., 1] = (f_vals[..., 0] * grad_f_vals[..., 1]) / (2.0 * half_extents[:, None])

    return grad
