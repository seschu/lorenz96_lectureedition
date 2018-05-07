#!/usr/bin/env python3
modulepath = '/scratch/local1/lorenz96_lectureedition/modules'
import sys
sys.path.append(modulepath)
from importlib import reload
import basics
import lorenz96 as model
import ginelli
reload(ginelli)
import numpy as np
import dill

######################
# Setting Parameters #
######################   

with open('ginelli_test', 'rb') as f:
    results = dill.load(f)


testbi=basics.test_bilinear(model.tendency,model.jacobian,results.dim,model.hessian,results.p, n=1)


CLVinv = np.memmap(results.expfolder +'/clvinv.dat',dtype = results.precision, order = results.order, shape = (len(results.time_mainrun), results.dim, results.dim), mode = "w+")
for it,t in enumerate(results.time_mainrun[:-1]):
    CLVinv[it,:,:] = np.linalg.inv(results.CLV[it,:,:])        
np.memmap.flush(CLVinv)

secondorder = np.memmap(results.expfolder +'/secondorder.dat',dtype = results.precision, order = results.order, shape = (len(results.time_mainrun), results.dim, results.dim), mode = "w+")


#
# optimize numpy einsum
#
it = 0
hessian = testbi.hessian(t,results.x[it,:],results.dt)
path_info,_ = np.einsum_path('ol,lik,ij,kj->oj',
          np.exp(CLVinv[it,:,:]),0.5*hessian[:,:,:],results.CLV[it,:,:],results.CLV[it,:,:], optimize='greedy')

for it,t in enumerate(results.time_mainrun[:-1]):
    print(it)
    hessian = testbi.hessian(t,results.x[it,:],results.dt)
    print(it)
    secondorder[it,:,:] = np.einsum('ol,lik,ij,kj->oj',
          np.exp(CLVinv[it,:,:]),0.5*hessian[:,:,:],results.CLV[it,:,:],results.CLV[it,:,:], optimize = path_info)

np.memmap.flush(secondorder)