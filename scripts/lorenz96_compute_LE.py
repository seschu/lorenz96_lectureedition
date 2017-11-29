#
# first lecture - Data Assimilation
# conceptual climate model
#

from lorenz96 import *
import numpy as np

######################
# Start Experiments  #
######################   
 
# run two experiments with different F
    
for F in [64]:    
    
    ######################
    # Setting Parameters #
    ######################   
    
    niceprint("Setting Parameters")
    
    # p is a dictionary
    p={'dimX' : 20,  # also called K
       'dimY' : 10,  # also called J
          'h' : 1,   # coupling strength
          'c' : 10,  # time scale 
          'b' : 10,  # length scale 
          'F' : F}   # forcing
    dt = 0.0001
    time_spinup = np.arange(0.0, 10.0, dt)
    time_mainrun = np.arange(0.0, 10.0, dt)

    # prepare 
        
    rk4tendency = rk4(L96_tendency)
    
    rk4tanlin = rk4_jacobian(L96_jacobian,L96_tendency,p['dimX']*p['dimY'] + p['dimX'])
    
    print(p)
    print("dt",dt)   
        
    #############################################
    # stationary state plus random perturbation #
    #############################################
    
    x0 = L96_stationary_state(p) # initial state (equilibrium)
    x0 = x0 + 0.1*np.hstack([np.random.rand(p['dimX'])*10,np.random.rand(p['dimX']*p['dimY'])]) # add small perturbation to 20th variable
    
        
    ######################
    #      Spin Up       #
    ######################
    
    niceprint("Spin Up")
    
    
    # setup time
    time = time_spinup
    
    # assign initial state
    y = x0
    
    # do integration in time
    
    
    for nstep ,( told , tnew ) in enumerate(zip(time[0:-1],time[1:])):
        print(nstep)
        y = y + rk4tendency(y,told,dt,p)
    
    # save inital state
    
    x0 = y
                    
    ######################
    #      Main Run      #
    ######################
    niceprint("Main Run")
    
    # setup time and result matrix
    
    time = time_mainrun
    x  = np.zeros((len(time), p['dimX']*p['dimY']+p['dimX']))
    
    LE = np.zeros((len(time), p['dimX']*p['dimY']+p['dimX']))
    
    
    # np.linalg.qr does the QR decomposition of a matrix.
    perturbations, _ =  np.linalg.qr(np.random.rand(p['dimX']*p['dimY']+p['dimX'],p['dimX']*p['dimY']+p['dimX']))
    
    y = x0
    # do integration in time
    for nstep ,( told , tnew ) in enumerate(zip(time[0:-1],time[1:])):
        print(nstep)
        x[nstep,:]=y    
        # multiply runge-kutta operator
        perturbations = perturbations + np.matmul(rk4tanlin(y,told,dt,p), perturbations)
        
        # step forward
        
        y = y + rk4tendency(y,told,dt, p)
        
        perturbations, r =  np.linalg.qr(perturbations)
        LE[nstep,:] = np.log(np.abs(np.diag(r)))/dt
        print(np.mean(LE[:nstep,0]))
    x[-1,:]=y    
        
    
    
    
    