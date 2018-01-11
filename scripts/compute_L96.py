#!/usr/bin/env python3
# -*- coding: utf-8 -*-

modulepath = 'C:\Users\Sebastian\Github\lorenz96_lectureedition\modules'
import sys
sys.path.append(modulepath)
from importlib import reload
import basics
import lorenz96 as model
import ginelli
reload(ginelli)
import numpy as np
import scipy as sp
import dill 

######################
# Start Experiments  #
######################   
 

######################
# Setting Parameters #
######################   

basics.niceprint("Setting Parameters")

# p is a dictionary
p={'dimX' : 8,  # also called K
   'dimY' : 32,  # also called J
      'h' : np.float64(0.1),   # coupling strength
      'c' : np.float64(10),  # time scale 
      'b' : np.float64(10),  # length scale 
      'F' : np.float64(8)}   # forcing

dt = 0.001
rescale_rate = dt*2
print(p)

dim = p['dimX']*p['dimY']+p['dimX']

time_spinup = np.arange(0.0, 0.01, dt*rescale_rate)
time_mainrun = np.arange(0.0, 0.01, dt*rescale_rate)

with open('test', 'wb') as f:
    dill.dump([time_spinup, time_mainrun,p,dt,rescale_rate], f)
#
#
#

#############################################
# stationary state plus random perturbation #
#############################################

x0 = model.stationary_state(p) # initial state (equilibrium)
x0 = x0 + np.linalg.norm(x0)*0.1*np.random.rand(dim)

GinelliL96 = ginelli.Run('test',model.tendency, model.jacobian,time_spinup,time_mainrun,x0,dim,p,rescale_rate,dt,'float64',memmap = True)
GinelliL96.ginelli()

GinelliL96.set_convergence_intervall(0,4)
GinelliL96.check_zero(zeromode=0)


import matplotlib.pyplot as plt
plt.figure();plt.plot(GinelliL96.zerocorr)
plt.figure();plt.plot(GinelliL96.cle_mean)


with open('ginelli_test', 'wb') as f:
    dill.dump(GinelliL96, f)
