#
# some modules for lorenz 96
#


# modules

import numpy as np
from itertools import product

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

from lorenz96_cython import L96_tendency as l96tend_c
def l96_tend_cw(t,X,p):
    dx = p['dimX']
    dy = p['dimY']    
    h  = p['h']
    c  = p['c']
    b  = p['b']
    return l96tend_c(0.0,X,dx,dy,h,c,b)    
    
jacobian = l96_tend_cw

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
    return np.array(np.bmat([[jacX_X,jacX_Y],
                     [jacY_X,jacY_Y]]))


from lorenz96_cython import L96_jacobian as l96c
def l96_cw(t,X,p):
    dx = p['dimX']
    dy = p['dimY']    
    h  = p['h']
    c  = p['c']
    b  = p['b']
    return l96c(0.0,X,dx,dy,h,c,b)    
    
jacobian = l96_cw






def L96_bilinear(t,a,b,para):
    
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
    
    
    tX =   np.einsum('ij,jk -> ijk',(a[:,XIp1] - a[:,XIm2]) , b[XIm1,:])      # Advection               
    tY = - para['c']*para['b']*np.einsum('ij,jk -> ijk',(a[:,YIp2] - a[:,YIm1]),b[YIp1,:])  # Advection
    
    return np.concatenate((tX,tY),axis=1)

bilinear = L96_bilinear

def L96_HessMat(t,x,para):
    
    dxy = para['dimY']*para['dimX']
    d = dxy + para['dimX']
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
    
    res = np.zeros((d,d,d),dtype=np.float64)
    for k in np.arange(0,para['dimX']):
        res[XI[k],XIp1[k],XIm1[k]]=1
        res[XI[k],XIm1[k],XIp1[k]]=1
        res[XI[k],XIm2[k],XIm1[k]]=-1
        res[XI[k],XIm1[k],XIm2[k]]=-1
    for k in np.arange(0,dxy):
        res[YI[k],YIp1[k],YIp2[k]]=-para['c']*para['b']
        res[YI[k],YIp2[k],YIp1[k]]=-para['c']*para['b']
        res[YI[k],YIp1[k],YIm1[k]]=para['c']*para['b']
        res[YI[k],YIm1[k],YIp1[k]]=para['c']*para['b']
    
    return res

hessian = L96_HessMat

      


def L96_stationary_state(para):
    
    x_stat = np.ones(para['dimY']*para['dimX']+para['dimX'])*para['F']
    
    return x_stat

    
stationary_state = L96_stationary_state


