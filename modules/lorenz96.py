#
# some modules for lorenz 96
#


# modules

import numpy as np


#########################
# L96 Tendency Function #
######################### 

    

def L96_tendency(t,X,para):
    
    ####################
    #      Indices     #
    ####################    
    
    XI = np.array(np.arange(0,para['dimX']), dtype = np.int)
    XIm1 = (XI - 1) % para['dimX']
    XIm2 = (XI - 2) % para['dimX']
    XIp1 = (XI + 1) % para['dimX']
    
    YI = np.arange(0,para['dimY']*para['dimX'])
    YIm1 = (YI - 1) % (para['dimY']*para['dimX']) + para['dimX']
    YIp2 = (YI + 2) % (para['dimY']*para['dimX']) + para['dimX']
    YIp1 = (YI + 1) % (para['dimY']*para['dimX']) + para['dimX']
    
    Ycoupling = np.array(np.floor(YI / para['dimY']),dtype=np.int)
    
    YI = YI  + para['dimX']
    
    ####################
    # X modes equation #
    ####################
               
    tX =  ((  X[XIp1] - X[XIm2]) * X[XIm1]      # Advection
            - X[XI]                             # Friction  
            + para['F']                         # Forcing
            - para['h']*para['c']/para['b'] *    # Coupling Parameter
            np.sum(X[para['dimX']:].reshape((para['dimY'],para['dimX']), order ='F'),axis = 0)) # Sum over the Y modes for each coupled X mode
    
    ####################
    # Y modes equation #
    ####################
    
    tY = ( - para['c']*para['b']*(X[YIp2] - X[YIm1]) * X[YIp1]  # Advection
           - para['c']*X[YI]                                    # Friction
           + para['h']*para['c']/para['b'] * X[Ycoupling])      # Coupling
    
    return np.hstack([tX,tY])   

tendency = L96_tendency


#########################
# L96 Jacobian Function #
######################### 

def L96_jacobian(t,X,para):
    
    dxy = para['dimY']*para['dimX']
    
    ####################
    #      Indices     #
    ####################   
        
    XI = np.arange(0,para['dimX'])
    XIm1 = (XI - 1) % para['dimX']
    XIm2 = (XI - 2) % para['dimX']
    XIp1 = (XI + 1) % para['dimX']
    
    YI = np.arange(0,dxy)
    YIm1 = ((YI - 1) % dxy) + para['dimX']
    YIp2 = ((YI + 2) % dxy) + para['dimX']
    YIp1 = ((YI + 1) % dxy) + para['dimX']   
    
    YI = YI  + para['dimX']
    
    mydiag = lambda a,k: np.roll(np.diag(a),shift=k,axis=1)
    jac_coupling = para['h']*para['c']/para['b']*np.array(np.matlib.repmat(np.hstack([np.ones(para['dimY']),np.zeros(dxy)]),1,para['dimX']))[0,:-dxy].reshape(para['dimX'],dxy,order ='C')
    
    
    #################################################
    # Differentiate X modes with respect to X modes #
    #################################################
    
    jacX_X = -np.eye(para['dimX']) + mydiag(X[XIp1]-X[XIm2],-1) + mydiag(X[XIm1],1) - mydiag(X[XIm1],-2)
    
    #################################################
    # Differentiate X modes with respect to Y modes #
    #################################################
    
    jacX_Y = - jac_coupling 

    #################################################
    # Differentiate Y modes with respect to X modes #
    #################################################

    jacY_X =  np.transpose(jac_coupling)

    #################################################
    # Differentiate Y modes with respect to X modes #
    #################################################

    jacY_Y = (  - para['c']*np.eye(dxy) 
                - para['c']*para['b']*(mydiag(X[YIp2]-X[YIm1],1) - mydiag(X[YIp1],-1) + mydiag(X[YIp1],2)))
    
#    return np.transpose(np.block([[jacX_X,jacX_Y],
#                     [jacY_X,jacY_Y]]))
    return np.bmat([[jacX_X,jacX_Y],
                     [jacY_X,jacY_Y]])


jacobian = L96_jacobian
    
def L96_stationary_state(para):
    
    x_stat = np.ones(para['dimY']*para['dimX']+para['dimX'])*para['F']
    
    return x_stat

    
stationary_state = L96_stationary_state