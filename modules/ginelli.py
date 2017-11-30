#
# fifth lecture - Lyapunov vectors
#


import numpy as np
import scipy as sp
import basics
import os

def startrun(expname,tendency,jacobian,time_spinup,time_mainrun,x0,dim,p,rescale_rate,dt,precision = 'float32'):
        
    ######################
    #      Ginelli       #
    ######################
    
    basics.niceprint("Ginelli")
    
    #
    # prepare functions of tendencies
    #
    
    rk4tendency = basics.rk4(tendency)
    
    rk4tendency ,rk4tanlin, rk4_jac_tend = basics.rk4_jacobian(jacobian,tendency, dim)
    
 
    #
    # 
    #
    expfolder = basics.root+"/runs/"+expname
    if not os.path.exists(expfolder ): os.mkdir(expfolder)
        
    ######################
    #      Spin Up       #
    ######################
    
    basics.niceprint("Spin Up")
    
    
    # setup time
    time = time_spinup
    
    # assign initial state
    y = x0
    
    # do integration in time
    
    for nstep ,( told , tnew ) in enumerate(zip(time[0:-1],time[1:])):
        basics.printProgressBar(nstep, len(time), prefix = 'Progress:', suffix = 'Complete', length = 20)
        for i in range(rescale_rate):
            y = y + rk4tendency(y,told,dt,p)
    
    # save inital state
    
    x0 = y
                    
    ######################
    #      Main Run      #
    ######################
    basics.niceprint("Main Run")
    
    # setup time and result matrix
    order = 'C'
     
    time = time_mainrun
    
    
    x  = np.memmap(expfolder +'/x.dat',dtype = precision, order = order, shape = (len(time), dim), mode ='w+')
    
    # define memory maps
    
    BLV = np.memmap(expfolder +'/blv.dat',dtype = precision, order = order, shape = (len(time), dim, dim), mode ='w+')
    
    BLE = np.memmap(expfolder +'/ble.dat',dtype = precision, order = order, shape = (len(time), dim), mode ='w+')
    
    CLV = np.memmap(expfolder +'/clv.dat',dtype = precision, order = order, shape = (len(time), dim, dim), mode ='w+')
    
    CLE = np.memmap(expfolder +'/cle.dat',dtype = precision, order = order, shape = (len(time), dim), mode ='w+')
    
    R = np.memmap(expfolder +'/r.dat',dtype = precision, order = order, shape = (len(time), dim, dim), mode ='w+')
    
    
    # initialize random orthogonal set of perturbations
    # np.linalg.qr does the QR decomposition of a matrix.
    BLV[0,:,:], _ =  np.linalg.qr(np.random.rand(dim,dim))
    
    #######################################
    #      Start Forward Computation      #
    #######################################
    basics.niceprint("Start Forward Computation")
    
    
    # initialize the non linear state
    y = x0
    # do integration in time
    stat =''
    for nstep ,( told , tnew ) in enumerate(zip(time[0:-1],time[1:])):
        basics.printProgressBar(nstep, len(time), prefix = 'Progress:', suffix = 'Complete; '+stat, length = 20)
        x[nstep,:]=y    
        
        # di timesteps with length dt for #rescale_rate repetitions
        V = BLV[nstep,:,:]
        for i in range(rescale_rate): 
            V, y = rk4_jac_tend(V, y,told,dt, p)
            
        BLV[nstep+1,:,:], R[nstep,:,:] =  np.linalg.qr(V)
        
        BLE[nstep,:] = np.log(np.abs(np.diag(R[nstep,:,:])))/(dt*rescale_rate)
        if nstep > 0: stat  = str(BLE[nstep,0]) + "   "+str(np.mean(BLE[nstep,0]))
        np.memmap.flush(R)
        np.memmap.flush(BLV)
        np.memmap.flush(BLE)
        np.memmap.flush(x) 
    x[-1,:]=y    
    np.memmap.flush(x) 
    
    #######################################
    #      Start Backward Computation     #
    #######################################
    basics.niceprint("Start Backward Computation")            
    
    # define function for dividing columns of a matrix by their respective norm
    
    def colnorm(M):
        norms = np.linalg.norm(M,axis=0,keepdims=True)
        return M/norms,norms
    
    # initialize random initial upper triangular matrix
    
    tri,_ = colnorm(sp.triu(np.random.rand(dim,dim)))
    
    for revnstep ,( tpast , tfuture ) in enumerate(zip(reversed(time[0:-1]),reversed(time[1:]))):
        nstep = len(time) - 1 - revnstep
        basics.printProgressBar(nstep, len(time), prefix = 'Progress:', suffix = 'Complete', length = 20)
        # solve R_{n,n+1}*X = C_{n+1}
        
        tri = sp.linalg.solve_triangular(R[nstep,:,:],tri)
        
        # normlize upper triangular matrix
        tri, growth = colnorm(sp.triu(np.random.rand(dim,dim)))
        
        # compute growth factor
        
        CLE[nstep,:] = -np.log(growth)/(dt*rescale_rate)
        
        # change from triangular representation to normal coordinates
        
        CLV[nstep,:,:]  = np.matmul(BLV[nstep,:,:],tri)
        
        np.memmap.flush(CLV)
        np.memmap.flush(CLE)
    basics.printProgressBar(nstep, len(time), prefix = 'Progress:', suffix = 'Complete', length = 20)
        
    return CLV,CLE,BLV,BLE