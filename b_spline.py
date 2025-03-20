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
    diff_scaled = diff / (2. * half_extents_reshaped) #scaled by the size of the node

    def f(x):
        #print(x)
        if x < 0:
            return f(-x)
        elif x >= 1.5:
            return 0.
        elif x >= 0.5:
            return (x - 1.5)**2 / 2.
        else:
            return - x**2 + 0.75
        
    #vals = np.maximum(0.5, - diff_scaled**2 + 0.75)  # shape (N, P, d)
    #vals = np.minimum(vals, (diff_scaled - 1.5)**2 / 2.)

    # 1D quadratic B-spline: f(t) = max(0, 1 - t^2)
    #vals = f(diff_scaled)  # shape (N, P, d)

    vals = np.zeros(diff_scaled.shape)
    for i in range(diff_scaled.shape[0]):
        for j in range(diff_scaled.shape[1]):
            vals[i,j,0] = f(diff_scaled[i,j,0])
            vals[i,j,1] = f(diff_scaled[i,j,1])

    return np.prod(vals, axis=2) #we multiply the two 1D B-splines to get the 2D B-spline

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
    diff_scaled = diff / (2. * half_extents_reshaped)  # (N, P, d)
    # Compute the 1D factors f(t) = max(0, 1 - t^2) for each dimension.

    def f(x):
        #print(x)
        if x < 0:
            return f(-x)
        elif x >= 1.5:
            return 0.
        elif x >= 0.5:
            return (x - 1.5)**2 / 2.
        else:
            return - x**2 + 0.75
    
    def grad_f(x):
        if x < 0:
            return - grad_f(-x)
        elif x >= 1.5:
            return 0.
        elif x >= 0.5:
            return x - 1.5
        else:
            return - 2 * x

    grad = np.zeros(diff_scaled.shape)

    for i in range(diff_scaled.shape[0]):
        for j in range(diff_scaled.shape[1]):
                grad[i,j,0] = grad_f(diff_scaled[i,j,0]) * f(diff_scaled[i,j,1]) / (2. * half_extents[i])
                grad[i,j,1] = f(diff_scaled[i,j,0]) * grad_f(diff_scaled[i,j,1]) / (2. * half_extents[i])
    return grad
