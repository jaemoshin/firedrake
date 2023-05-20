import numpy as np
from firedrake import *
from data_gen import gauge_set
from data_fit import pcn
from solver import solve_tides
from solver import gauge_settwo
# Define the parameters for gauge_set
c = Constant(0.001)
t_trunc = 900
gauge_num = 20
nsteps = 1200

TideSolver, wn, wn1, = solve_tides(c = Constant(0.001))

# Call the gauge_set function
result = gauge_settwo(TideSolver, wn, wn1, c=c, t_trunc=t_trunc, gauge_num=gauge_num, nsteps=nsteps)

print(result)
# Import the pcn function from data_fit
from data_fit import pcn

# Define the parameters for pcn

iterations = 100
beta = 1/2
cov = np.ones((1, 1))

# Call the pcn function
pcn_result = pcn(TideSolver, wn, wn1, result, c=Constant(0.01), iter=iterations, beta=beta, cov=cov, nsteps=nsteps)

# Print or process the pcn_result as needed
print(pcn_result)