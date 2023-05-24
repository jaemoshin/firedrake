from firedrake import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

mesh = PeriodicRectangleMesh(50,50,20000, 5000, direction = "x") #mesh with actual length
V = FunctionSpace(mesh, "BDM", 1) #Linear vector fields in each triangle
#normal component is continuous between triangles
Q = FunctionSpace(mesh, "DG", 0) #constant in each triangle
#no continuity constraints
W = V*Q 
wn = Function(W) 
wn1 = Function(W)  

v, phi = TestFunctions(W)
f = Constant(10**-5) #10^-5 Coriolis?
#f = Constant(0)
g = Constant(9.81) #9.81 #gravity constant
#b = Constant(0) #0 Sea Level
b = Function(Q)  
x,y = SpatialCoordinate(mesh)
midx = Constant(10000)
midy = Constant(2500)
scale = Constant(1000)
b.interpolate(350*exp( -((x-midx)**2 / scale**2/2 + (y-midy)**2 / scale**2/ 2) )) #Gaussian Hill
c = Constant(0.001) #? Damping Constant (unknown value)
dt0 = 12*3600/50 
dt = Constant(dt0) #12*3600/50 timestep
H = Constant(700) #700 Ocean depth
t = Constant(0) #time
#F0 = Constant(10**-7) #10^-7 
F0 = 0
F = F0*as_vector((sin(2*pi*t/(12*3600)), 0)) 

un, etan = wn.split()
print('norm before', norm(etan))
un, etan = split(wn)
un1, etan1 = split(wn1)
unh = (un + un1)/2
etanh = (etan + etan1)/2
equation = (
    inner(v, un1 - un) + f*inner(v, as_vector((-unh[1], unh[0])))*dt
    - g*div(v)*(etanh - b)*dt
    - inner(F, v)*dt
    + c*inner(v, unh)*dt
    + phi*(etan1 - etan) + H*phi*div(unh)*dt
)*dx

Bc = [DirichletBC(W.sub(0), [0,0], "on_boundary")]
TideProblem = NonlinearVariationalProblem(equation, wn1, bcs = Bc)
solver_parameters = {
    'mat_type': 'matfree',
    'pc_type': 'python',
    'pc_python_type': 'firedrake.HybridizationPC',
    'ksp_type': 'preonly',
    #'ksp_monitor_true_residual': True,
    'hybridization': {
        'ksp_type': 'cg',
        #'ksp_converged_reason':None,
        'ksp_rtol': 1e-6,
        'pc_type': 'lu',
        'pc_gamg_sym_graph': None,
        'mg_levels': {
            'ksp_type': 'chebyshev',
            'ksp_chebyshev_esteig': None,
            'ksp_max_it': 5,
            'pc_type': 'bjacobi',
            'sub_pc_type': 'ilu'
        }
    }
}

TideSolver = NonlinearVariationalSolver(TideProblem, solver_parameters=solver_parameters)

t0 = 0.0
file0 = File("tide.pvd")
u, eta = wn.split()
file0.write(u, eta)

nsteps = 2500
list = np.zeros(nsteps)

for step in range(nsteps):
    t0 += dt0
    t.assign(t0)
    TideSolver.solve()
    print(norm(eta))
    wn.assign(wn1)
    if step%10 == 0:
        file0.write(u, eta)
    print(t0)
    list[step] = eta.at(0.5,0.5) #sample at this point
 
np.savetxt('loceta.txt', list)
#with open(r'/home/jms19/M4R/tides/loceta.txt', 'w') as fp:
#    fp.write("%s\n" % list)

#xaxis = np.arange(10)
#print(list)
#plt.plot(list)