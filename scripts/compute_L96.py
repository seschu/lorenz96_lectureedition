#!/usr/bin/env python3
# -*- coding: utf-8 -*-

modulepath = '/scratch/uni/u234/u234069/lorenz96_lectureedition/modules'
import sys
sys.path.append(modulepath)

import basics
import lorenz96 as l96
import ginelli
import numpy as np
import scipy as sp
import dill 

######################
# Start Experiments  #
######################   
 
model = l96

######################
# Setting Parameters #
######################   

basics.niceprint("Setting Parameters")

# p is a dictionary
p={'dimX' : 20,  # also called K
   'dimY' : 10,  # also called J
      'h' : 0.2,   # coupling strength
      'c' : 10,  # time scale 
      'b' : 10,  # length scale 
      'F' : 10}   # forcing
dt = 0.00001
rescale_rate = 100
print(p)

dim = p['dimX']*p['dimY']+p['dimX']

time_spinup = np.arange(0.0, 0.0, dt*rescale_rate)
time_mainrun = np.arange(0.0, 0.2, dt*rescale_rate)

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

CLV,CLE,BLV,BLE = ginelli.startrun('test',model.tendency, model.jacobian,time_spinup,time_mainrun,x0,dim,p,rescale_rate,dt,'float64')