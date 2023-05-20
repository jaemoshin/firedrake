import numpy as np
from firedrake import *
from solver import gauge_settwo

def pcn(TideSolver, y_act, c = Constant(0.01), iter = 10, beta = 1/2, cov = np.ones((1,1)), nsteps = 1200):

  import numpy as np
  import matplotlib.pyplot as plt
  """
  inputs

  n: an integer which represents the number of iterations
  c
  beta: Weight; number between 0 and 1 
  cov: a nxn covariance matrix

  output

  c
  """

  def phi(y_act, y_obs):
    """
    Return phi value
    """
    entries = y_act-y_obs
    inv_cov = np.identity(entries.shape[0])/np.var(entries)
    return np.sum(entries.T@inv_cov@entries)

  len = 1
  J = np.log(c.dat.data)[0]
  acc_probs = []
  exp_J_hats = []
  cumulative_avg = np.exp(J)

  for k in ProgressBar(f'iterations').iter(range(iter)):
    print(k)
    xi = np.random.multivariate_normal(np.zeros(( len, )) + np.log(0.01), cov , size = len)#Centred Gaussian Measure
    #positive J ~ multivariate normal (log c0, )
    #c = exp(J) 
    #generate both c from the same distribution
    #
    J_hat = np.sqrt(1 - beta**2)*J + beta*xi[0][0]

    unif = np.random.uniform(0,1) 
    
    y_obs_c = gauge_settwo(TideSolverone, Constant(np.exp(J)), t_trunc = 900, gauge_num = 20, nsteps = nsteps)

    TideSolvertwo = TideSolver.solve()
    y_obs_c_hat = gauge_settwo(TideSolvertwo, Constant(np.exp(J_hat)), t_trunc = 900, gauge_num = 20, nsteps = nsteps)

    acc_prob = np.minimum(1, np.exp(phi(y_act, y_obs_c) - phi(y_act, y_obs_c_hat)))
    
    print(y_act)
    print(y_obs_c)
    print(y_obs_c_hat)

    
    acc_probs.append(acc_prob)
    exp_J_hats.append(np.exp(J_hat))
    
    if unif <= acc_prob:
       J = J_hat
       cumulative_avg = (cumulative_avg * k + np.exp(J_hat))/(k + 1)

  # Save the results to a file
  np.savetxt('pcn_results.txt', np.array([np.exp(J)]), fmt='%.6f')
  np.savetxt('acc_probs.txt', np.array(acc_probs), fmt='%.6f')
  np.savetxt('exp_J_hats.txt', np.array(exp_J_hats), fmt='%.6f')
  np.savetxt('cumulative_avg.txt', np.array([cumulative_avg]), fmt='%.6f')
  return np.exp(J), acc_probs, exp_J_hats, cumulative_avg
