#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 12:10:30 2017

@author: u234069
"""
import numpy as np

p = {'dimX' : 20,  # also called K
    'dimY' : 10,  # also called J
    'h' : np.float64(0.2),   # coupling strength
    'c' : np.float64(10),  # time scale 
    'b' : np.float64(10),  # length scale 
    'F' : np.float64(10)}   # forcing
 
dx = p['dimX']
dy = p['dimY']

h  = p['h']
c  = p['c']
b  = p['b']
F  = p['F']

dim = dx*dy+dx

test = np.random.rand(dim)

import lorenz96_cython as l96c
import lorenz96 as model
j =  l96c.L96_jacobian(0.0,test,dx,dy,h,c,b)
j2 =  model.jacobian(0,test,p)

import matplotlib.pyplot as plt
print(np.max(np.abs(j-j2)))


t =  l96c.L96_tendency(0.0,test,dx,dy,h,c,b,F)
t2 =  model.tendency(0,test,p)
print(np.max(np.abs(t-t2)))

import basics
rk4tendency_c ,rk4tanlin_c, rk4_jac_tend_c = basics.rk4_constructor(model.jacobian,model.tendency, dim)
rk4tendency ,rk4tanlin, rk4_jac_tend = basics.rk4_constructor(model.L96_jacobian,model.L96_tendency, dim)


V = np.random.rand(dim,dim)
y = np.random.rand(dim)

V1, y1 = rk4_jac_tend(V, y,0,1e-5,p)
V2, y2 = rk4_jac_tend_c(V, y,0,1e-5,p)

y1 = rk4tendency( y,0,1e-5,p)
y2 = rk4tendency_c(y,0,1e-5,p)

V1 = rk4tanlin( test,0,1e-5,p)
V2 = rk4tanlin_c(test,0,1e-5,p)

                
def l96_v2(t,X,p):
    dx = p['dimX']
    dy = p['dimY']    
    h  = p['h']
    c  = p['c']
    b  = p['b']
    return l96c.L96_jacobian(0.0,test,dx,dy,h,c,b)



"""
import lorenz96 as model

%timeit model.jacobian(0,test,p)

%timeit l96c.L96_jacobian(0.0,test,dx,dy,h,c,b)



1st 1.32 ms ± 21.5 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
2nd 826 µs ± 19.3 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)


"""