#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

cimport numpy as np
from libc.math cimport floor
cimport cython


#########################
# L96 Jacobian Function #
######################### 
mydiag = lambda a,k: np.roll(np.diag(a),shift=k,axis=1)
#  """
#
# tX =  ((  X[XIp1] - X[XIm2]) * X[XIm1]      # Advection
#            - X[XI]                             # Friction  
#            + para['F']                         # Forcing
#            - para['h']*para['c']/para['b'] *    # Coupling Parameter
#            np.sum(X[para['dimX']:].reshape((para['dimY'],para['dimX']), order ='F'),axis = 0)) # Sum over the Y modes for each coupled X mode
#    
#    ####################
#    # Y modes equation #
#    ####################
#    
#    tY = ( - para['c']*para['b']*(X[YIp2] - X[YIm1]) * X[YIp1]  # Advection
#           - para['c']*X[YI]                                    # Friction
#           + para['h']*para['c']/para['b'] * X[Ycoupling])      # Coup
#
#"""  


@cython.boundscheck(False)
@cython.wraparound(False)
def rk4_constructor(jac,f,dim):

    jac1 = lambda t,y,p,dt : jac(t,y,p)
    jac2 = lambda t,y,p,dt,k1 : jac(t,y + dt/2*k1,p)
    jac3 = lambda t,y,p,dt,k2 : jac(t,y + dt/2*k2,p)
    jac4 = lambda t,y,p,dt,k3 : jac(t,y + dt*k3,p)
    
    def func_rk4(y,t,dt,p):
        k1res = f(t,y,p)
        k2res = f(t,y + dt/2*k1res,p)
        k3res = f(t,y + dt/2*k2res,p)
        k4res = f(t,y + dt*k3res,p)
        return dt/6*(k1res + 2*k2res +2*k3res + k4res)
    
    def jac_rk4(y,t,dt,p):
        
        k1res = f(t,y,p)
        k2res = f(t,y + dt/2*k1res,p)
        k3res = f(t,y + dt/2*k2res,p)
        
        jac2res = jac2(t,y,p,dt,k1res)
        jac3res = jac3(t,y,p,dt,k2res)
        jac4res = jac4(t,y,p,dt,k3res)
        
        deriv_k1 = jac1(t,y,p,dt)
        deriv_k2 = jac2res + np.matmul(jac2res,dt/2*deriv_k1)
        deriv_k3 = jac3res + np.matmul(jac3res,dt/2*deriv_k2)
        deriv_k4 = jac4res + np.matmul(jac4res,dt  *deriv_k3) 
        return dt/6*(deriv_k1 + 2* deriv_k2 + 2* deriv_k3 + deriv_k4)
    
    def all_rk4(V,y,t,dt,p):
        
        k1res = f(t,y,p)
        k2res = f(t,y + dt/2*k1res,p)
        k3res = f(t,y + dt/2*k2res,p)
        k4res = f(t,y + dt*k3res,p)
        
        jac2res = jac2(t,y,p,dt,k1res)
        jac3res = jac3(t,y,p,dt,k2res)
        jac4res = jac4(t,y,p,dt,k3res)
        
        deriv_k1 = jac1(t,y,p,dt)
        deriv_k2 = jac2res + np.matmul(jac2res,dt/2*deriv_k1)
        deriv_k3 = jac3res + np.matmul(jac3res,dt/2*deriv_k2)
        deriv_k4 = jac4res + np.matmul(jac4res,dt  *deriv_k3)
        
        
        return V + dt/6*np.matmul((deriv_k1 + 2* deriv_k2 + 2* deriv_k3 + deriv_k4), V), y + dt/6*(k1res + 2 * k2res + 2* k3res + k4res)
    
    
    return func_rk4, jac_rk4, all_rk4


