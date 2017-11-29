#
# fifth lecture - Lyapunov vectors
#

from lorenz96 import *
import numpy as np
import scipy as sp
import dill 

precision ='float32'

######################
# Start Experiments  #
######################   
 
# run two experiments with different F#
    
for ie, F in enumerate([64]):
    
    ######################
    # Setting Parameters #
    ######################   
    
    niceprint("Setting Parameters")
    
    # p is a dictionary
    p={'dimX' : 20,  # also called K
       'dimY' : 10,  # also called J
          'h' : 0.2,   # coupling strength
          'c' : 10,  # time scale 
          'b' : 10,  # length scale 
          'F' : F}   # forcing
    dt = 0.00001
    rescale_rate = 1000
    
    dim = p['dimX']*p['dimY']+p['dimX']
    
    time_spinup = np.arange(0.0, 10.0, dt*rescale_rate)
    time_mainrun = np.arange(0.0, 10.0, dt*rescale_rate)
    
    with open('exp'+str(ie).zfill(4), 'wb') as f:
        dill.dump([time_spinup, time_mainrun,p,dt,rescale_rate], f)
    #
    #
    #
    
    
    
    # prepare 
        
    rk4tendency = rk4(L96_tendency)
    
    rk4tanlin, rk4_jac_tend = rk4_jacobian(L96_jacobian,L96_tendency, dim)
    
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
        for i in range(rescale_rate):
            y = y + rk4tendency(y,told,dt,p)
    
    # save inital state
    
    x0 = y
                    
    ######################
    #      Main Run      #
    ######################
    niceprint("Main Run")
    
    # setup time and result matrix
    order = 'C'
     
    time = time_mainrun
    x  = np.memmap('x.dat',dtype = precision, order = order, shape = (len(time), dim), mode ='w+')
    
    # define memory maps
    
    BLV = np.memmap('blv.dat',dtype = precision, order = order, shape = (len(time), dim, dim), mode ='w+')
    
    BLE = np.memmap('ble.dat',dtype = precision, order = order, shape = (len(time), dim), mode ='w+')
    
    CLV = np.memmap('clv.dat',dtype = precision, order = order, shape = (len(time), dim, dim), mode ='w+')

    CLE = np.memmap('cle.dat',dtype = precision, order = order, shape = (len(time), dim), mode ='w+')
    
    R = np.memmap('r.dat',dtype = precision, order = order, shape = (len(time), dim, dim), mode ='w+')
    
    
    # initialize random orthogonal set of perturbations
    # np.linalg.qr does the QR decomposition of a matrix.
    BLV[0,:,:], _ =  np.linalg.qr(np.random.rand(dim,dim))
    
    #######################################
    #      Start Forward Computation      #
    #######################################
    niceprint("Start Forward Computation")
    
    
    # initialize the non linear state
    y = x0
    # do integration in time
    for nstep ,( told , tnew ) in enumerate(zip(time[0:-1],time[1:])):
        print(nstep)
        x[nstep,:]=y    
        
        # di timesteps with length dt for #rescale_rate repetitions
        V = BLV[nstep,:,:]
        for i in range(rescale_rate): 
#            # multiply runge-kutta operator
#            V = V + np.matmul(rk4tanlin(y,told,dt,p), V)
#            # step forward
#            y = y + rk4tendency(y,told,dt, p)
            V, y = rk4_jac_tend(V, y,told,dt, p)
            
        BLV[nstep+1,:,:], R[nstep,:,:] =  np.linalg.qr(V)
        
        BLE[nstep,:] = np.log(np.abs(np.diag(R[nstep,:,:])))/(dt*rescale_rate)
        print(BLE[nstep,0], np.mean(BLE[:nstep,0]))
    x[-1,:]=y    
    
    #######################################
    #      Start Backward Computation     #
    #######################################
    niceprint("Start Backward Computation")            
    
    # define function for dividing columns of a matrix by their respective norm
    
    def colnorm(M):
        norms = np.linalg.norm(M,axis=0,keepdims=True)
        return M/norms,norms
    
    # initialize random initial upper triangular matrix
    
    tri,_ = colnorm(sp.triu(np.random.rand(dim,dim)))
    
    for nstep ,( tpast , tfuture ) in enumerate(zip(reversed(time[0:-1]),reversed(time[1:]))):
        
        # solve R_{n,n+1}*X = C_{n+1}
        
        tri = sp.linalg.solve_triangular(R[nstep,:,:],tri)
        
        # normlize upper triangular matrix
        tri, growth = colnorm(sp.triu(np.random.rand(dim,dim)))
        
        # compute growth factor
        
        CLE[nstep,:] = -np.log(growth)/(dt*rescale_rate)
        
        # change from triangular representation to normal coordinates
        
        CLV[nstep,:,:]  = np.matmul(BLV[nstep,:,:],tri)
        