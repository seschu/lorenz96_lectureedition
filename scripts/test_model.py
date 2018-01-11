#
# test linear and bilienar functions and errorscaling
#

modulepath = 'C:\Users\Sebastian\Github\lorenz96_lectureedition\modules'
import sys
sys.path.append(modulepath)
from importlib import reload
import basics
reload(basics)
import lorenz96 as model
reload(model)
import numpy as np

######################
# Start Experiments  #
######################   
 

######################
# Setting Parameters #
######################   

basics.niceprint("Setting Parameters")

# p is a dictionary
p={'dimX' : 20,  # also called K
   'dimY' : 10,  # also called J
      'h' : np.float64(0.2),   # coupling strength
      'c' : np.float64(10),  # time scale 
      'b' : np.float64(10),  # length scale 
      'F' : np.float64(10)}   # forcing

dim = p['dimX']*p['dimY']+p['dimX']

print(p)

#test_euler=basics.test_jac(model.tendency,model.jacobian,dim,model.hessian,p,epsrange=10.**np.arange(-10,1.1,0.1),n=1)
#test_euler.do_test(use_odeint=False,scale =np.float64(10))
#fig1 = test_euler.show_scaling(deg = 1,epsfitrange=[10.0**-1,10**2])



dts = 10.**np.arange(-6,-3,1)
epsrange = 10.**np.arange(-10,1,1)

x  = np.zeros((len(epsrange),len(dts)))
yj = np.zeros((len(epsrange),len(dts)))
yh = np.zeros((len(epsrange),len(dts)))


import matplotlib.pyplot as plt
        
testbi=basics.test_bilinear(model.tendency,model.jacobian,dim,model.hessian,p,epsrange = epsrange, n=1)
for i,dt in enumerate(dts):
    testbi.do_test(T = 0.001, dt=dt,scale =np.float64(1), sparse = False)
    f=testbi.show_scaling(deg = 1,epsfitrange=10.**np.arange(-10,1.1,1))
    plt.close(f)
    x[:,i], yj[:,i], yh[:,i] = testbi.get_plot_data()
    

plt.figure();
for i,dt in enumerate(dts):
    pl=plt.plot(x[:,i],yj[:,i],linestyle=':',label='_nolegned_')
    plt.plot(x[:,i],yh[:,i],color=pl[0].get_c(),linestyle='-',label="dt = "+str(dt))
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid()
plt.xlabel('initial error norm')
plt.ylabel('norm of error')
plt.title("rk4: dotted Line 1st order, solid line 1st and 2nd order")
#plt.savefig('errorscaling.png',dpi =300) 
#plt.savefig('errorscaling.pdf')    


