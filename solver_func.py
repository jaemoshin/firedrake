from firedrake import *
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

def solve_tides(c):
    """
    Create and return a solver for the tide simulation problem.

    inputs:
    c: Probability distribution (Function or Expression)

    output:
    TideSolver: NonlinearVariationalSolver object
    """
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
    
    mesh = PeriodicRectangleMesh(50, 50, 20000, 5000, direction="x")  # mesh with actual length
    V = FunctionSpace(mesh, "BDM", 1)  # Linear vector fields in each triangle
    Q = FunctionSpace(mesh, "DG", 0)  # constant in each triangle
    W = V * Q
    wn = Function(W)
    wn1 = Function(W)

    v, phi = TestFunctions(W)
    f = Constant(10 ** -5)  # 10^-5 Coriolis?
    g = Constant(9.81)  # 9.81 gravity constant
    x, y = SpatialCoordinate(mesh)
    midx = Constant(10000)
    midy = Constant(2500)
    scale = Constant(1000)
    b = Function(Q)
    b.interpolate(350 * exp(-((x - midx) ** 2 / scale ** 2 / 2 + (y - midy) ** 2 / scale ** 2 / 2)))  # Gaussian Hill

    # Smooth c by solving a Poisson problem
    c_smooth_data = smooth_continuous_distribution(c, iterations=10, seed=123)[1]
    c_smooth = Function(Q)
    c_smooth.dat.data[:] = c_smooth_data

    dt0 = 12 * 3600 / 50
    dt = Constant(dt0)  # 12*3600/50 timestep
    H = Constant(700)  # 700 Ocean depth
    t = Constant(0)  # time
    F0 = Constant(10 ** -1)  # 10^-7
    F = F0 * as_vector((sin(2 * pi * t / (12 * 3600)), 0))

    un, etan = wn.split()
    un, etan = split(wn)
    un1, etan1 = split(wn1)
    unh = (un + un1) / 2
    etanh = (etan + etan1) / 2

    equation = (
        inner(v, un1 - un) + f * inner(v, as_vector((-unh[1], unh[0]))) * dt
        - g * div(v) * (etanh) * dt
        - inner(F, v) * dt
        + c_smooth * inner(v, unh) * dt
        + phi * (etan1 - etan) + (H - b) * phi * div(unh) * dt
    ) * dx

    Bc = [DirichletBC(W.sub(0), [0, 0], "on_boundary")]
    TideProblem = NonlinearVariationalProblem(equation, wn1, bcs=Bc)
    solver_parameters = {
        'snes_lag_jacobian': 1,
        'snes_lag_preconditioner': 1,
        'snes_max_it': 10,
        'snes_atol': 1e-6,
        'snes_rtol': 1e-6,
        'snes_monitor': None,
        'snes_converged_reason': None,
    }
    TideSolver = NonlinearVariationalSolver(TideProblem, solver_parameters=solver_parameters)
    
    return TideSolver, wn, wn1, t, F0, c

def gauge_settwo(TideSolver, wn, wn1, t, t_trunc = 900, gauge_num = 20, nsteps = 1200):
    
        
    t0 = 0.0
    dt0 = 12*3600/50 
    file0 = File("tide.pvd")
    u, eta = wn.split()
    
    listt = np.zeros((gauge_num, nsteps))
    
    wn.assign(0)
    wn1.assign(0)

    for step in ProgressBar(f'nsteps').iter(range(nsteps)):
        t0 += dt0
        t.assign(t0)
        TideSolver.solve()
        #print(norm(eta))
        wn.assign(wn1)
        #if step%10 == 0:
            #file0.write(u, eta)
        #print(t0)
        
        for j in range(gauge_num):

            listt[j][step] = eta.at(j*0.1+ 0.5,0.5) #sample at this point
    
    
    array_2d = np.array(listt[:, t_trunc:])
    vector = array_2d.flatten()
    return vector
