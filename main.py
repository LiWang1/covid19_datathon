"""
Created on Wed Apr 15 11:35:56 2020

@author: wangli
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import scipy.optimize as opt



# read the data
S = [990, 900, 600, 200, 100, 0  ]  # susceptible people 
I = [10,  100, 300, 600, 600, 600]  # infected people 
R = [0,     0, 100, 200, 300, 400]  # recoverd people 
obs = np.transpose(np.array([S, I, R]))


# set up initial conditions 
S0 = 990 
I0 = 10
R0 = 0 
init = [S0,I0,R0]
days = 5

def pred(beta, gamma): 
    # beta:  transmit rate 
    # gamma: recover rate 

    sir0 = init # initial condition  
    
    N = sum(init) # total population 
    
    # time span 
    t = np.linspace(0,days,days+1) 
    
    # SIR model 
    def sir_model(sir,t):
        dsdt = - beta*sir[0]*sir[1]/N
        didt = beta*sir[0]*sir[1]/N - gamma*sir[1]
        drdt = gamma*sir[1]
        dsirdt = [dsdt, didt, drdt]
        return dsirdt

    # solve 
    z = odeint(sir_model,sir0,t)
    return z

# loss function
def loss(x): 
    prediction = pred(x[0], x[1])
    mal = np.sum(np.abs(np.subtract(prediction, obs))) # mean absolute error 
    return mal
    
# optimization 
x0 = [0.9, 0.1]
res = opt.minimize(loss, x0, method = 'BFGS')

# prediction 
z = pred(res.x[0], res.x[1])
tspan = np.linspace(0,days,days+1) 

# plot 
plt.plot(tspan,z[:,0],'b-',label=r's')
plt.plot(tspan,S, 'b--', label=r's_obs')
plt.plot(tspan,z[:,1],'r-',label=r'i')
plt.plot(tspan,I, 'r--', label=r'i_obs')
plt.plot(tspan,z[:,2],'g-',label=r'r')
plt.plot(tspan,R, 'g--', label=r'r_obs')
plt.ylabel('response')
plt.xlabel('time')
plt.legend(loc='best')
plt.show()