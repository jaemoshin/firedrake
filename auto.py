import numpy as np
from firedrake import *
from data_gen import gauge_set
from data_fit import pcn

# Define the parameters for gauge_set
c = Constant(0.001)
t_trunc = 900
gauge_num = 20
nsteps = 1200

# Call the gauge_set function
result = gauge_set(c=c, t_trunc=t_trunc, gauge_num=gauge_num, nsteps=nsteps)

# Import the pcn function from data_fit
from data_fit import pcn

# Define the parameters for pcn
y_act = result  # Use the result from gauge_set
c = 0.01
iterations = 3
beta = 1/2
cov = np.ones((1, 1))
nsteps = 1200

# Call the pcn function
pcn_result = pcn(y_act, c=c, iter=iterations, beta=beta, cov=cov, nsteps=nsteps)

# Print or process the pcn_result as needed
print(pcn_result)
