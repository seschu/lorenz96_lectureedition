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
def L96_tendency(t,np.ndarray[double, ndim=1, mode='c'] X not None,dx,dy,h,c,b,F):
    
    cdef np.intp_t xi,yi,dxy 
    cdef np.ndarray[double, ndim=1, mode='c'] tendency= np.zeros((dx+dx*dy,))
    
    dxy = dy*dx
    
    
    xi=0
    tendency[xi] =      (X[xi + 1] - X[dx - 2]) * X[dx - 1] - X[xi] + F - h*c/b*np.sum(X[dx + dy*xi:dx + dy*(xi+1)])
    
    xi=1
    tendency[xi] =      (X[xi + 1] - X[dx - 1]) * X[0] - X[xi] + F - h*c/b*np.sum(X[dx + dy*xi:dx + dy*(xi+1)])
    
    for xi in range(2,dx-1):
        tendency[xi] =      (X[xi + 1] - X[xi - 2]) * X[xi - 1] - X[xi] + F - h*c/b*np.sum(X[dx + dy*xi:dx + dy*(xi+1)])
        
    xi=dx-1
    tendency[xi] =      (X[0] - X[xi-2]) * X[xi - 1] - X[xi] + F - h*c/b*np.sum(X[dx + dy*xi:dx + dy*(xi+1)])
        
    yi = dx
    tendency[yi] = -c*b*(X[yi + 2] - X[dx+dxy - 1]) * X[yi + 1] - c*X[yi] + h*c/b*X[int(floor((yi-dx)/dy))]
    
    for yi in range(dx+1,dx+dxy-2):
        tendency[yi] = -c*b*(X[yi + 2] - X[yi - 1]) * X[yi + 1] - c*X[yi] + h*c/b*X[int(floor((yi-dx)/dy))]
    
    yi = dx+dxy-2
    tendency[yi] = -c*b*(X[dx] - X[yi - 1]) * X[yi + 1] - c*X[yi]  + h*c/b*X[int(floor((yi-dx)/dy))]
    
    yi = dx+dxy-1
    tendency[yi] = -c*b*(X[dx + 1] - X[yi - 1]) * X[dx] - c*X[yi]  + h*c/b*X[int(floor((yi-dx)/dy))]
    
    
    return tendency


@cython.boundscheck(False)
@cython.wraparound(False)
def L96_jacobian(t,np.ndarray[double, ndim=1, mode='c'] X not None,dx,dy,h,c,b):
    
    cdef np.intp_t xi,yi,dxy 
    cdef np.ndarray[double, ndim=2, mode='c'] jacobian = np.zeros((dx+dx*dy, dx+dx*dy))
    ####################
    #      Indices     #
    ####################   
        
    dxy = dy*dx
    
    jacobian[0,0] = -1
    jacobian[0,dx-1] = X[1]-X[dx-2]
    jacobian[0,1] = X[dx-1]
    jacobian[0,dx-2] = -X[dx-1]
    for yi in range(dx ,dx + dy):
        jacobian[0,yi] = - h*c/b

    jacobian[1,1] = -1
    jacobian[1,0] = X[2]-X[dx-1]
    jacobian[1,2] = X[0]
    jacobian[1,dx-1] = -X[0]
    for yi in range(dx + dy,dx + 2*dy):
        jacobian[1,yi] = - h*c/b

    
    for xi in range(2,dx-1):
        
        jacobian[xi,xi] = -1
        jacobian[xi,xi-1] = X[xi+1]-X[xi-2]
        jacobian[xi,xi+1] = X[xi-1]
        jacobian[xi,xi-2] =-X[xi-1]
        for yi in range(dx + xi*dy,dx + (xi +1)*dy):
            jacobian[xi,yi] = - h*c/b


    jacobian[dx-1,dx-1] = -1
    jacobian[dx-1,dx-2] = X[0]-X[dx-3]
    jacobian[dx-1,0] = X[dx-2]
    jacobian[dx-1,dx-3] = -X[dx-2]
    for yi in range(dx + (dx-1)*dy,dx + dx*dy):
        jacobian[dx-1,yi] = - h*c/b


    yi = dx
    jacobian[yi,yi] = -c
    jacobian[yi,yi+1] = -c*b*(X[yi+2]-X[dx*dy+dx-1])
    jacobian[yi,yi+2] = -c*b*X[yi+1]
    jacobian[yi,dx*dy+dx-1] =  c*b*X[yi+1]
    jacobian[yi,0] =  h*c/b

    
    for yi in range(dx+1,dx+dxy-2):
        
        jacobian[yi,yi]   = -c
        jacobian[yi,yi+1] = -c*b*(X[yi+2]-X[yi-1])
        jacobian[yi,yi+2] = -c*b*X[yi+1]
        jacobian[yi,yi-1] =  c*b*X[yi+1]
        xi = int(floor((yi-dx)/dy))
        jacobian[yi,xi] =  h*c/b

    yi = dx+dxy-2
    jacobian[yi,yi] = -c
    jacobian[yi,yi+1] = -c*b*(X[dx]-X[yi-1])
    jacobian[yi,dx  ]   = -c*b*X[yi+1]
    jacobian[yi,yi-1] =  c*b*X[yi+1]
    xi = int(floor((yi-dx)/dy))
    jacobian[yi,xi] =  h*c/b

    yi = dx+dxy-1
    jacobian[yi,yi] = -c
    jacobian[yi,dx] =   -c*b*(X[dx+1]-X[yi-1])
    jacobian[yi,dx+1] = -c*b*X[dx]
    jacobian[yi,yi-1] =  c*b*X[dx]
    xi = int(floor((yi-dx)/dy))
    jacobian[yi,xi] =  h*c/b

    return jacobian

def rk4(t,np.ndarray[double, ndim=1, mode='c'] X not None,dt,dx,dy,h,c,b,F):
    
    cdef np.ndarray[double, ndim=1, mode='c'] k1 = np.zeros((dx+dx*dy))
    cdef np.ndarray[double, ndim=1, mode='c'] k2 = np.zeros((dx+dx*dy))
    cdef np.ndarray[double, ndim=1, mode='c'] k3 = np.zeros((dx+dx*dy))
    cdef np.ndarray[double, ndim=1, mode='c'] k4 = np.zeros((dx+dx*dy))
    
    k1 = L96_tendency(t,X,dx,dy,h,c,b,F)
    k2 = L96_tendency(t,X + dt/2*k1,dx,dy,h,c,b,F)
    k3 = L96_tendency(t,X + dt/2*k2,dx,dy,h,c,b,F)
    k4 = L96_tendency(t,X + dt  *k3,dx,dy,h,c,b,F)
    
    return dt/6*(k1+2.0*k2+2.0*k3 + k4)