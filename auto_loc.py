import numpy as np
from firedrake import *
import matplotlib.pyplot as plt
from solve_loc import solve_tides, gauge_settwo

plt.clf()
TideSolver, wn, wn1, t, F0, c = solve_tides(Constant(0.001))
res = gauge_settwo(TideSolver, wn, wn1, t, 0, 20, 1000)
resultthree = res.reshape((20,1000))
plt.plot(resultthree[0,:])
c = c.dat.data[0]
F0 = F0.dat.data[0]
plt.savefig(f"c{c}F{F0}.pdf")


