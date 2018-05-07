#
# fifth lecture - Lyapunov vectors
#


import numpy as np
import scipy as sp
import basics
import os

class Run():
    def __init__(self,expname,tendency,jacobian,time_spinup,time_mainrun,x0,dim,p,rescale_rate,dt, existing = False, memmap = True, precision = 'float64'):
        
        # experiment folder defined
        self.expname = expname
        self.expfolder = basics.root+"/runs/"+self.expname
        self.existing = existing
        if not os.path.exists(self.expfolder): 
            if not existing: os.mkdir(self.expfolder)
            else: ValueError("No experiment with this name exists.")
        
        writemode = 'r+' if existing else 'w+'
        
        # define memory maps
        self.order = 'F'
        if memmap:
            self.x  = np.memmap(self.expfolder +'/x.dat',dtype = precision, order = self.order, shape = (len(time_mainrun), dim), mode = writemode)
            
            self.BLV = np.memmap(self.expfolder +'/blv.dat',dtype = precision, order = self.order, shape = (len(time_mainrun), dim, dim), mode = writemode)
            
            self.BLE = np.memmap(self.expfolder +'/ble.dat',dtype = precision, order = self.order, shape = (len(time_mainrun), dim), mode = writemode)
            
            self.CLV = np.memmap(self.expfolder +'/clv.dat',dtype = precision, order = self.order, shape = (len(time_mainrun), dim, dim), mode = writemode)
            
            self.CLE = np.memmap(self.expfolder +'/cle.dat',dtype = precision, order = self.order, shape = (len(time_mainrun), dim), mode = writemode)
            
            self.R = np.memmap(self.expfolder +'/r.dat',dtype = precision, order = self.order, shape = (len(time_mainrun), dim, dim), mode = writemode)
        else:
            self.x  = np.zeros((len(time_mainrun), dim))
            
            self.BLV = np.zeros((len(time_mainrun), dim, dim))
            
            self.BLE = np.zeros((len(time_mainrun), dim))
            
            self.CLV = np.zeros((len(time_mainrun), dim, dim))
            
            self.CLE = np.zeros((len(time_mainrun), dim))
            
            self.R = np.zeros((len(time_mainrun), dim, dim))

        self.expname = expname
        self.tendency = tendency
        self.jacobian = jacobian
        self.time_spinup = time_spinup
        self.time_mainrun = time_mainrun
        self.x0 = x0
        self.dim = dim
        self.p = p
        self.rescale_rate = np.int(rescale_rate)
        self.dt = dt
        self.precision = precision
        
        self.rk4tendency ,self.rk4tanlin, self.rk4_jac_tend = basics.rk4_constructor(jacobian,tendency, dim)
    
    def ginelli(self):
        self.forward()
        self.backward()
    def forward(self):        
        ######################
        #      Ginelli       #
        ######################
        
        basics.niceprint("Ginelli")
        
            
        ######################
        #      Spin Up       #
        ######################
        
        basics.niceprint("Spin Up")
        
        
        # setup time
        time = self.time_spinup
        time_init = np.arange(0,self.dt,self.dt/10)
        # assign initial state
        y = self.x0
        
        # do integration in time
        
        for nstep ,( told , tnew ) in enumerate(zip(time_init[0:-1],time_init[1:])):
            basics.printProgressBar(nstep, len(time_init)+len(time)-2, prefix = 'Progress:', suffix = 'Complete', length = 20)
            for i in range(0,self.rescale_rate):
                y = y + self.rk4tendency(y,told,self.dt,self.p)
        
        
        for nstep ,( told , tnew ) in enumerate(zip(time[0:-1],time[1:])):
            basics.printProgressBar(nstep + len(time_init), len(time_init)+len(time)-2, prefix = 'Progress:', suffix = 'Complete', length = 20)
            for i in range(self.rescale_rate):
                y = y + self.rk4tendency(y,told,self.dt,self.p)
        
        # save inital state
        
        self.x1 = y
                        
        ######################
        #      Main Run      #
        ######################
        basics.niceprint("Main Run")
        
        # setup time and result matrix
         
        time = self.time_mainrun
        
        
    
        
        
        # initialize random orthogonal set of perturbations
        # np.linalg.qr does the QR decomposition of a matrix.
        self.BLV[0,:,:], _ =  np.linalg.qr(np.random.rand(self.dim,self.dim))
        
        #######################################
        #      Start Forward Computation      #
        #######################################
        basics.niceprint("Start Forward Computation")
        
        
        # initialize the non linear state
        y = self.x1
        # do integration in time
        stat =''
        for nstep ,( told , tnew ) in enumerate(zip(time[0:-1],time[1:])):
            basics.printProgressBar(nstep, len(time), prefix = 'Progress:', suffix = 'Complete; '+stat, length = 20)
            self.x[nstep,:]=y    
            
            # di timesteps with length dt for #rescale_rate repetitions
            V = self.BLV[nstep,:,:]
            for i in range(self.rescale_rate): 
                V, y = self.rk4_jac_tend(V, y,told,self.dt,self.p)
                
            self.BLV[nstep+1,:,:], self.R[nstep,:,:] =  np.linalg.qr(V)
            
            self.BLE[nstep,:] = np.log(np.abs(np.diag(self.R[nstep,:,:])))/(self.dt*self.rescale_rate)
            if nstep > 0: stat  = str(self.BLE[nstep,0]) + "   "+str(np.mean(self.BLE[:nstep,0]))
            if nstep % 50 == 0:
                np.memmap.flush(self.R)
                np.memmap.flush(self.BLV)
                np.memmap.flush(self.BLE)
                np.memmap.flush(self.x) 
        self.x[-1,:]=y    
        np.memmap.flush(self.x) 
        
        np.memmap.flush(self.R)
        np.memmap.flush(self.BLV)
        np.memmap.flush(self.BLE)
        np.memmap.flush(self.x) 
        
        
    #
    # Perform Ginelli backward operation
    #
    def backward(self):
        #######################################
        #      Start Backward Computation     #
        #######################################
        basics.niceprint("Start Backward Computation")            
        
        # define function for dividing columns of a matrix by their respective norm
        
        def colnorm(M):
            norms = np.linalg.norm(M,axis=0,keepdims=True)
            return M/norms,norms
        
        # initialize random initial upper triangular matrix
        
        tri,_ = colnorm(sp.triu(np.random.rand(self.dim,self.dim)))
        
        for revnstep ,( tpast , tfuture ) in enumerate(zip(reversed(self.time_mainrun[0:-1]),reversed(self.time_mainrun[1:]))):
            nstep = len(self.time_mainrun) - 1 - revnstep
            basics.printProgressBar(nstep, len(self.time_mainrun), prefix = 'Progress:', suffix = 'Complete', length = 20)
            # solve R_{n,n+1}*X = C_{n+1}
            
            tri = sp.linalg.solve_triangular(self.R[nstep-1,:,:],tri)
            
            # normlize upper triangular matrix
            tri, growth = colnorm(tri)
            
            # compute growth factor
            
            self.CLE[nstep-1,:] = -np.log(growth)/(self.dt*self.rescale_rate)
            
            # change from triangular representation to normal coordinates
            
            self.CLV[nstep-1,:,:]  = np.matmul(self.BLV[nstep-1,:,:],tri)
            
            np.memmap.flush(self.CLV)
            np.memmap.flush(self.CLE)
        basics.printProgressBar(nstep, len(self.time_mainrun), prefix = 'Progress:', suffix = 'Complete', length = 20)
        
    def check_zero(self,zeromode = 0):
        #
        # This function finds smallest LE and looks at correlation with the zero exponent
        #
        if zeromode == 0:
            self.zeromode = np.abs(self.get_cle()).argmin()
            zeromode = self.zeromode
        else:
            self.zeromode = zeromode
        self.zerocorr = np.memmap(self.expfolder +'/zerocorr.dat',dtype = self.precision, order = self.order, shape = (len(self.time_mainrun),), mode = 'w+')
        time = self.time_mainrun
        for nstep ,( told , tnew ) in enumerate(zip(time[0:-1],time[1:])):
            zeroLV = self.CLV[nstep,:,self.zeromode]
            tendency = self.rk4tendency(self.x[nstep,:],told,self.dt,self.p)
            self.zerocorr[nstep] = np.abs(np.sum(np.multiply(zeroLV,tendency/np.linalg.norm(tendency))))
            
    
    def set_convergence_intervall(self,begin,end):
        #
        # This function
        #
        
        self.a = int(begin/(self.dt*self.rescale_rate))
        self.b = int(end/(self.dt*self.rescale_rate))
        
    def get_cle(self):
        self.cle_mean = np.memmap.mean(self.CLE[self.a:self.b,:], axis = 0)
        return self.cle_mean
        