#
# some modules for lorenz 96
#


# modules

import numpy as np
from scipy.integrate import odeint
#import pandas as pd
from scipy.integrate import ode
import pandas as pd

def rk4(f):
    return lambda y, t, dt, p: (
            lambda dy1: (
            lambda dy2: (
            lambda dy3: (
            lambda dy4: (dy1 + 2*dy2 + 2*dy3 + dy4)/6
            )( dt * f( t + dt,y + dy3 , p  ) )
	    )( dt * f( t + dt/2, y + dy2/2, p) )
	    )( dt * f( t + dt/2, y + dy1/2, p) )
	    )( dt * f( t        , y       , p ) )

def rk4_jacobian(jac,f,dim):

    k1 = lambda t,y,p,dt : f(t,y,p)
    k2 = lambda t,y,p,dt : f(t,y + dt/2*k1(t,y,p,dt),p)
    k3 = lambda t,y,p,dt : f(t,y + dt/2*k2(t,y,p,dt),p)
    k4 = lambda t,y,p,dt : f(t,y + dt*k3(t,y,p,dt),p)
    
    jac1 = lambda t,y,p,dt : jac(t,y,p)
    jac2 = lambda t,y,p,dt,k1 : jac(t,y + dt/2*k1,p)
    jac3 = lambda t,y,p,dt,k2 : jac(t,y + dt/2*k2,p)
    jac4 = lambda t,y,p,dt,k3 : jac(t,y + dt*k3,p)
    
    def jac_rk4(y,t,dt,p):
        
        jac1res = jac1(t,y,p,dt)
        jac2res = jac2(t,y,p,dt)
        jac3res = jac3(t,y,p,dt)
        jac4res = jac4(t,y,p,dt)
        
        deriv_k2 = jac2res + np.matmul(jac2res,dt/2*jac1res )
        deriv_k3 = jac3res + np.matmul(jac3res,dt/2*deriv_k2)
        deriv_k4 = jac4res + np.matmul(jac4res,dt  *deriv_k3) 
        return dt/6*(jac1res + 2* deriv_k2 + 2* deriv_k3 + deriv_k4)
    
    def all_rk4(V,y,t,dt,p):
        
        k1res = k1(t,y,p,dt)
        k2res = k2(t,y,p,dt)
        k3res = k3(t,y,p,dt)
        k4res = k4(t,y,p,dt)
        
        jac1res = jac1(t,y,p,dt)
        jac2res = jac2(t,y,p,dt,k1res)
        jac3res = jac3(t,y,p,dt,k2res)
        jac4res = jac4(t,y,p,dt,k3res)
        
        deriv_k2 = jac2res + np.matmul(jac2res,dt/2*jac1res )
        deriv_k3 = jac3res + np.matmul(jac3res,dt/2*deriv_k2)
        deriv_k4 = jac4res + np.matmul(jac4res,dt  *deriv_k3)
        
        return V + dt/6*np.matmul((jac1res + 2* deriv_k2 + 2* deriv_k3 + deriv_k4), V), y + dt/6*(k1res + 2 * k2res + 2* k3res + k4res)
    
    return jac_rk4, all_rk4




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



#########################
# L96 Jacobian Function #
######################### 

def L96_jacobian(t,X,para):
    
    dxy = para['dimY']*para['dimX']
    d = para['dimY']*para['dimX'] + para['dimX']
    
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


def L96_stationary_state(para):
    
    x_stat = np.ones(para['dimY']*para['dimX']+para['dimX'])*para['F']
    
    return x_stat

def twist(f):
    return lambda a,b,*args:f(b,a,*args)


def niceprint(text,width = 25,level = 0):
    
    if level == 0:
        topline="#"
        lowline="#"
        intro="#"
        outro="#"
    elif level ==1:
        topline=""
        lowline="-"
        intro="  "
        outro="  "
    elif level ==2:
        topline=""
        lowline=""
        intro="    "
        outro="    "
        
    text=text.replace("\n","\\n")
    text=text.split()
    wordlengths = list(map(len,text))
    width = np.max([width,max(wordlengths)])+1
    linelength = wordlengths[0]
    formatedstring = intro + "  " + text[0]
    
    for i, (length, word) in enumerate(zip(wordlengths[1:],text[1:])):
        numberoflines = 0
        if word == "\\n":
            formatedstring = formatedstring + " "*(width-linelength) + "  "+outro+"\n"        
            linelength = 0
            formatedstring = formatedstring + intro+" "
        else:
            if length + linelength < width:
                formatedstring = formatedstring + " " + word
                if linelength > 0: linelength = linelength + 1
                linelength = length + linelength
            else:
                before = int(np.floor((width - linelength)/2))
                after = int(np.ceil((width - linelength)/2))
                beginline = numberoflines*(width + len(intro)+len(outro)+4+1)+len(intro)+2
                dummy = formatedstring[:beginline]
                formatedstring = dummy + " "*before + formatedstring[beginline:] + " "*after + "  "+outro+"\n"        
                numberoflines =+ 1
                linelength = length 
                formatedstring = formatedstring + intro+"  " + word
    
    formatedstring = formatedstring + " "*(width-linelength) + "  "+outro+"\n"
    formatedstring = topline*(width+4+len(outro)+len(intro)) +"\n" + formatedstring + lowline*(width+4+len(outro)+len(intro)) 
    print(formatedstring)
    