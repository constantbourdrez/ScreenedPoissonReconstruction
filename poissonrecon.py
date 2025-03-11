import numpy as np
import matplotlib.pyplot as plt
from normals import compute_local_PCA
from ply import write_ply, read_ply
from scipy.interpolate import BSpline

# Import the point clouds 
path = ''
points = read_ply(path)

# Import the normals
normals = compute_local_PCA(points, points)

# Define the grid 

# Define the B-spline basis functions
splines = []
k = 2
knots = np.array([])
n_basis = len(knots) - (k + 1)
for i in range(n_basis):
    # Create B-spline basis function
    coeffs = np.zeros(n_basis)
    coeffs[i] = 1  # Activate one basis function at a time
    spline = BSpline(knots, coeffs, k)
    splines.append(spline)

# Solve the linear system