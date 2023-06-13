import numpy as np
from firedrake import *
from data_fit_func import pcn
from solver_func import solve_tides
from solver_func import gauge_settwo
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

mesh = PeriodicRectangleMesh(50, 50, 20000, 5000, direction="x")
V = FunctionSpace(mesh, "BDM", 1)
Q = FunctionSpace(mesh, "DG", 0)  # constant in each triangle
# PCG64 random number generator
pcg = PCG64(seed=123456789)
rg = RandomGenerator(pcg)
# normal distribution
f_normal = rg.normal(V, 0.0, 1.0)

def smooth_continuous_distribution(data, iterations=10, seed=None):
    np.random.seed(seed)
    n = len(data)
    rhs = np.random.rand(n)
    smoothed_dist = data.copy()

    for _ in range(iterations):
        laplacian = diags([-1, 2, -1], [-1, 0, 1], shape=(n, n), format='csr')
        smoothed_dist = spsolve(laplacian, rhs, permc_spec='NATURAL')
        rhs = smoothed_dist + np.random.rand(n)

    integral = np.trapz(smoothed_dist, data)  # Normalize the distribution
    if integral != 0:
        smoothed_dist /= integral
    return data, smoothed_dist

c = smooth_continuous_distribution(f_normal.dat.data, iterations=10, seed=123)[1]

print(len(c))