import numpy as np
from firedrake import *
from data_fit_func import pcn
from solver_func import solve_tides
from solver_func import gauge_settwo
# Define the parameters for gauge_set

c_min = 0.00005
c_max = 0.00015
c_expression = np.linspace(c_min, c_max, num=5000)

t_trunc = 0
gauge_num =  20
nsteps = 100
TideSolver, wn, wn1, t, F0, c = solve_tides(c_expression)

# Call the gauge_set function

result = gauge_settwo(TideSolver, wn, wn1, t= t, t_trunc=t_trunc, gauge_num=gauge_num, nsteps=nsteps)

print(result)
"""
# Import the pcn function from data_fit
from data_fit_func import pcn

# Define the parameters for pcn

iterations = 1
beta = 0.05
cov = np.ones((1, 1))

# Call the pcn function
pcn_result = pcn(TideSolver, wn, wn1, t, result, c=Constant(0.001), iter=iterations, beta=beta, cov=cov, t_trunc = t_trunc, nsteps=nsteps)

# Print or process the pcn_result as needed
print(pcn_result)"""