#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

root = "/scratch/local1/lorenz96_lectureedition"

def twist(f):
    return lambda a,b,*args:f(b,a,*args)

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()


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
        
        dim = V.shape[0]
        k1res = f(t,y,p)
        k2res = f(t,y + dt/2*k1res,p)
        k3res = f(t,y + dt/2*k2res,p)
        k4res = f(t,y + dt*k3res,p)
        
        deriv_k1 = jac(t,y,p)
        deriv_k2 = np.matmul(jac(t,y + dt/2*k1res,p),np.eye(dim) + dt/2*deriv_k1)
        deriv_k3 = np.matmul(jac(t,y + dt/2*k2res,p),np.eye(dim) + dt/2*deriv_k2)
        deriv_k4 = np.matmul(jac(t,y + dt*k3res,p),np.eye(dim) + dt  *deriv_k3)
        
        return V + dt/6*np.matmul((deriv_k1 + 2* deriv_k2 + 2* deriv_k3 + deriv_k4), V), y + dt/6*(k1res + 2 * k2res + 2* k3res + k4res)
    
    
    return func_rk4, jac_rk4, all_rk4

def rk4_bilinear_constructor(hessian,jac,f,dim):
    jac1 = lambda t,y,p,dt : jac(t,y,p)
    jac2 = lambda t,y,p,dt,k1 : jac(t,y + dt/2*k1,p)
    jac3 = lambda t,y,p,dt,k2 : jac(t,y + dt/2*k2,p)
    jac4 = lambda t,y,p,dt,k3 : jac(t,y + dt*k3,p)
    
    
    if callable(hessian):
        def bilinear_rk4(y,t,dt,p):
            
            k1res = f(t,y,p)
            k2res = f(t,y + dt/2*k1res,p)
            k3res = f(t,y + dt/2*k2res,p)
            
            jac2res = jac2(t + dt/2,y,p,dt,k1res)
            jac3res = jac3(t + dt/2,y,p,dt,k2res)
            jac4res = jac4(t + dt  ,y,p,dt,k3res)
            
            deriv_k1 = jac1(t,y,p,dt)
            deriv_k2 = np.matmul(jac2res,np.eye(dim) + dt/2*deriv_k1)
            deriv_k3 = np.matmul(jac3res,np.eye(dim) + dt/2*deriv_k2)
            
            h1 = hessian(t,y,p)
            h2 = np.einsum('iml,mk,lj', hessian(t + dt/2,y + dt/2*k1res,p), np.eye(dim) + dt/2*deriv_k1, np.eye(dim) + dt/2*deriv_k1) + dt/2 * np.einsum( 'il,lkj', jac2res, h1)
            h3 = np.einsum('iml,mk,lj', hessian(t + dt/2,y + dt/2*k2res,p), np.eye(dim) + dt/2*deriv_k2, np.eye(dim) + dt/2*deriv_k2) + dt/2 * np.einsum( 'il,lkj', jac3res, h2)
            h4 = np.einsum('iml,mk,lj', hessian(t + dt  ,y + dt  *k3res,p), np.eye(dim) + dt  *deriv_k3, np.eye(dim) + dt  *deriv_k3) + dt   * np.einsum( 'il,lkj', jac4res, h3)
            
            return dt/6*(h1 + 2 * h2 + 2 * h3 + h4)
    
    else:
        def bilinear_rk4(y,t,dt,p):
            
            k1res = f(t,y,p)
            k2res = f(t,y + dt/2*k1res,p)
            k3res = f(t,y + dt/2*k2res,p)
            
            jac2res = jac2(t,y,p,dt,k1res)
            jac3res = jac3(t,y,p,dt,k2res)
            jac4res = jac4(t,y,p,dt,k3res)
            
            deriv_k1 = jac1(t,y,p,dt)
            deriv_k2 = np.matmul(jac2res,np.eye(dim) + dt/2*deriv_k1)
            deriv_k3 = np.matmul(jac3res,np.eye(dim) + dt/2*deriv_k2)
            
#            print('h1')
#            h1 = hessian
#            print('h2')
#            h2 = np.einsum('iml,mk,lj', hessian, np.eye(dim) + dt/2*deriv_k1, np.eye(dim) + dt/2*deriv_k1) + dt/2 * np.einsum( 'il,lkj', jac2res, h1)
#            print('h3')
#            h3 = np.einsum('iml,mk,lj', hessian, np.eye(dim) + dt/2*deriv_k2, np.eye(dim) + dt/2*deriv_k2) + dt/2 * np.einsum( 'il,lkj', jac3res, h2)
#            print('h4')
#            h4 = np.einsum('iml,mk,lj', hessian, np.eye(dim) + dt  *deriv_k3, np.eye(dim) + dt  *deriv_k3) + dt   * np.einsum( 'il,lkj', jac4res, h3)
#            
            print('h1')
            h1 = hessian
            print('h2')
            h2 = np.einsum('iml,mk,lj', hessian, np.eye(dim) + dt/2*deriv_k1, np.eye(dim) + dt/2*deriv_k1) + dt/2 * np.einsum( 'il,lkj', jac2res, h1)
            print('h3')
            h3 = np.einsum('iml,mk,lj', hessian, np.eye(dim) + dt/2*deriv_k2, np.eye(dim) + dt/2*deriv_k2) + dt/2 * np.einsum( 'il,lkj', jac3res, h2)
            print('h4')
            h4 = np.einsum('iml,mk,lj', hessian, np.eye(dim) + dt  *deriv_k3, np.eye(dim) + dt  *deriv_k3) + dt   * np.einsum( 'il,lkj', jac4res, h3)
            
            return dt/6*(h1 + 2 * h2 + 2 * h3 + h4)
        
        Ihessian,Vhessian = sparse_detect(hessian)
        Ihessian,Vhessian = sparse_detect(hessian)
        
        def bilinear_rk4_sparse(y,t,dt,p):
            
            k1res = f(t,y,p)
            k2res = f(t,y + dt/2*k1res,p)
            k3res = f(t,y + dt/2*k2res,p)
            
            jac2res = jac2(t,y,p,dt,k1res)
            jac3res = jac3(t,y,p,dt,k2res)
            jac4res = jac4(t,y,p,dt,k3res)
            
            deriv_k1 = jac1(t,y,p,dt)
            deriv_k2 = np.matmul(jac2res,np.eye(dim) + dt/2*deriv_k1)
            deriv_k3 = np.matmul(jac3res,np.eye(dim) + dt/2*deriv_k2)
            
#            print('h1')
#            h1 = hessian
#            print('h2')
#            h2 = np.einsum('iml,mk,lj', hessian, np.eye(dim) + dt/2*deriv_k1, np.eye(dim) + dt/2*deriv_k1) + dt/2 * np.einsum( 'il,lkj', jac2res, h1)
#            print('h3')
#            h3 = np.einsum('iml,mk,lj', hessian, np.eye(dim) + dt/2*deriv_k2, np.eye(dim) + dt/2*deriv_k2) + dt/2 * np.einsum( 'il,lkj', jac3res, h2)
#            print('h4')
#            h4 = np.einsum('iml,mk,lj', hessian, np.eye(dim) + dt  *deriv_k3, np.eye(dim) + dt  *deriv_k3) + dt   * np.einsum( 'il,lkj', jac4res, h3)
#            
            h1 = hessian
            h2 = hessian_biproduct_sparse(Ihessian,Vhessian,dim, np.eye(dim) + dt/2*deriv_k1, np.eye(dim) + dt/2*deriv_k1) + dt/2 * hessian_mult( jac2res, h1, dim) #hessian_mult_sparse(Ihessian,Vhessian,dim, jac2res)
            h3 = hessian_biproduct_sparse(Ihessian,Vhessian,dim, np.eye(dim) + dt/2*deriv_k2, np.eye(dim) + dt/2*deriv_k2) + dt/2 * hessian_mult( jac3res, h2, dim)#hessian_mult_sparse(Ihessian2,Vhessian2,dim, jac3res)
            h4 = hessian_biproduct_sparse(Ihessian,Vhessian,dim, np.eye(dim) + dt  *deriv_k3, np.eye(dim) + dt  *deriv_k3) + dt   * hessian_mult( jac4res, h3, dim)#hessian_mult_sparse(Ihessian3,Vhessian3,dim, jac4res)
            
            return dt/6*(h1 + 2 * h2 + 2 * h3 + h4)
    
    return bilinear_rk4, bilinear_rk4_sparse


from itertools import product

def sparse_detect(mat,threshold=0.0):
    list_of_indices = []
    values = []
    for K in product(*list(map(lambda x:range(0,x),mat.shape))):
        if np.abs(mat[K]) > threshold:
            list_of_indices.append(K)
            values.append(mat[K])
            
    return list_of_indices, values

def construct_dense(list_of_indices, values,mat):
    
    for i,K in enumerate(list_of_indices):
        mat[K] = values[i]
            
    return mat


def hessian_biproduct_sparse(hi,hv,idim,a,b):
    res = np.zeros((idim,idim,idim),dtype = np.float64)
    for i,K in enumerate(hi):
        res[K[0],:,:] = res[K[0],:,:] + hv[i]*a[K[1],:,np.newaxis]*b[K[2],:]
    return res

def hessian_biproduct_sparse_vec(hi,hv,idim,a,b):
    res = np.zeros(idim,dtype = np.float64)
    for i,K in enumerate(hi):
        res[K[0]] = res[K[0]] + hv[i]*a[K[1]]*b[K[2]]
    return res

def hessian_biproduct_sparse_vec_notconst(hi,mat,idim,a,b):
    res = np.zeros(idim,dtype = np.float64)
    for i,K in enumerate(hi):
        res[K[0]] = res[K[0]] + mat[K]*a[K[1]]*b[K[2]]
    return res


def hessian_mult(a,h,idim):
    res = np.zeros((idim,idim,idim),dtype = np.float64)
    for i in range(0,idim):
        res[:,:,i] = np.matmul(a,h[:,:,i])
    return res

class test_jac():
    import numpy as np
    import matplotlib.pyplot as plt
            
    """
    Test errorscaling of jacobian to a tendency function
    """
    def __init__(self,tend,jac,dim,hess = None ,para=None,n=100,time=0,epsrange = 10**np.arange(-8,1.1),tendtimeord=0,jactimeord=0):
        self.tend = lambda t,x: tend(t,x,para)
        self.jac = lambda t,x: jac(t,x,para)
        self.dim = dim
        self.n = n
        self.time = time
        self.epsrange = epsrange
        self.tendtimeord = tendtimeord
        self.jactimeord = jactimeord
        self.testdone = False
        self.scalar = lambda x,y : np.sum(np.dot(x.reshape(1,-1),y.reshape(-1,1)), dtype = np.float64 )
        self.norm = lambda x : np.sqrt(self.scalar(x,x), dtype = np.float64)
        self.normalize = lambda x : x/self.norm(x)
        self.corr = lambda x,y: self.scalar(self.normalize(x),self.normalize(y))
        self.hess = lambda t,x: hess(t,x,para)
        
    def do_test(self, scale = 1, use_odeint = False, dt = 0.001, rtol = None, atol = None):
        
        if use_odeint:           
            def tend(t,x):
                if self.tendtimeord==0: 
                    intres = odeint(lambda x,t : self.tend(t,x), x, [0, dt],  Dfun = self.jac,rtol=rtol,atol=atol, mxstep=10**5)
                else:
                    intres = odeint(self.tend, x, [0, dt],  Dfun = self.jac,rtol=rtol,atol=atol, mxstep=10**5)
                return (intres[-1,:] - intres[-2,:])/dt
        else:
            tend = self.tend
        self.correlation_jac = np.zeros((len(self.epsrange)),dtype = np.float64)
        self.error_jac = np.zeros((len(self.epsrange)),dtype = np.float64)
        self.correlation_hess = np.zeros((len(self.epsrange)),dtype = np.float64)
        self.error_hess = np.zeros((len(self.epsrange)),dtype = np.float64)
        
        self.jacdiff = np.zeros((len(self.epsrange)),dtype = np.float64)
        self.hessdiff= np.zeros((len(self.epsrange)),dtype = np.float64)
        
        for ieps,eps in enumerate(self.epsrange):
            for i in range(0, self.n):
                testa = np.float64(self.normalize(np.random.rand(self.dim)))*scale
                testb = np.float64(testa+eps*self.normalize(np.random.rand(self.dim)))
                
                if self.tendtimeord==0:  tenddiff = (tend(self.time,testb) - tend(self.time,testa))
                else:               tenddiff = (tend(testb,self.time) - tend(testa,self.time)) 
                
                if self.jactimeord==0: jacdiff = np.matmul(self.jac(self.time,testa),(testb-testa))
                else:             jacdiff = np.matmul(self.jac(testa,self.time),(testb-testa))
                
                if self.jactimeord==0: 
                    hessdiff = (jacdiff 
                                + np.float64(1/2)*np.einsum('...ij,i,j',self.hess(self.time,testa),testb-testa,testb-testa , dtype=np.float64))
                else:
                    hessdiff = (jacdiff 
                                + np.float64(1/2)*np.einsum('...ij,i,j',self.hess(testa,self.time),testb-testa,testb-testa , dtype=np.float64))
                
                
                self.correlation_jac[ieps]  = self.correlation_jac[ieps] + np.corrcoef(tenddiff,jacdiff)[0,1]
                self.error_jac[ieps]        = self.error_jac[ieps] + self.norm(tenddiff-jacdiff)
                self.correlation_hess[ieps] = self.correlation_hess[ieps] + np.corrcoef(tenddiff,hessdiff)[0,1]
                self.error_hess[ieps]       = self.error_hess[ieps] + self.norm(tenddiff-hessdiff)
                self.jacdiff[ieps]          = self.jacdiff[ieps] + self.norm(jacdiff)
                self.hessdiff[ieps]         = self.hessdiff[ieps] + self.norm(hessdiff-jacdiff)
                
            self.error_jac[ieps]        = self.error_jac[ieps]          /np.float64(self.n)
            self.correlation_hess[ieps] = self.correlation_hess[ieps]   /np.float64(self.n)
            self.error_hess[ieps]       = self.error_hess[ieps]         /np.float64(self.n)
            self.jacdiff[ieps]          = self.jacdiff[ieps]            /np.float64(self.n)
            self.hessdiff[ieps]         = self.hessdiff[ieps]           /np.float64(self.n)
        self.testdone = True
    
    def show_scaling(self, epsfitrange = None, epsplotrange = None, deg = 1):
        if isinstance(epsfitrange, type(None)): epsfitrange = np.array([self.epsrange.min(), self.epsrange.max()])
        if not isinstance(epsfitrange,np.ndarray): 
            epsfitrange = np.array(epsfitrange)
        self.epsfitrange = epsfitrange
        if isinstance(epsplotrange, type(None)): epsplotrange = np.array([self.epsrange.min(), self.epsrange.max()])
        if not isinstance(epsplotrange,np.ndarray): 
            epsplotrange = np.array(epsplotrange)
        self.epsplotrange = epsplotrange
        if self.testdone:
            x = self.epsrange[(self.epsrange>=epsfitrange.min()) & (self.epsrange<=epsfitrange.max())]
            y = self.error_jac[(self.epsrange>=epsfitrange.min()) & (self.epsrange<=epsfitrange.max())]
            
            fig = plt.figure()
            self.logx = np.log(np.float64(x))
            self.logy = np.log(np.float64(y))
            self.coeffs = np.polyfit(self.logx,self.logy,deg=deg,full = True)
            poly = np.poly1d(self.coeffs[0])
            yfit = lambda x: np.exp(poly(self.logx))
            
            xplot = self.epsrange[(self.epsrange>=epsplotrange.min()) & (self.epsrange<=epsplotrange.max())]
            yplot = self.error_jac[(self.epsrange>=epsplotrange.min()) & (self.epsrange<=epsplotrange.max())]
            
            self.fitting_jac = np.round(self.coeffs[0][0],2)
            
            plt.plot(xplot,yplot,linestyle='None',marker='o',markerfacecolor = 'b')
            plt.plot(np.float64(x),yfit(np.float64(x)),color ='k',linestyle=':')
            
            x = self.epsrange[(self.epsrange>=epsfitrange.min()) & (self.epsrange<=epsfitrange.max())]
            y = self.error_hess[(self.epsrange>=epsfitrange.min()) & (self.epsrange<=epsfitrange.max())]
            
            self.logx = np.log(np.float64(x))
            self.logy = np.log(np.float64(y))
            self.coeffs = np.polyfit(self.logx,self.logy,deg=deg,full = True)
            self.fitting_hess = np.round(self.coeffs[0][0],2)
            poly = np.poly1d(self.coeffs[0])
            yfit = lambda x: np.exp(poly(self.logx))
            
            xplot = self.epsrange[(self.epsrange>=epsplotrange.min()) & (self.epsrange<=epsplotrange.max())]
            yplot = self.error_hess[(self.epsrange>=epsplotrange.min()) & (self.epsrange<=epsplotrange.max())]
            
            plt.plot(xplot,yplot,linestyle='None',marker='o',markerfacecolor = 'k')
            plt.plot(x,yfit(x),color ='k',linestyle=':')
            plt.xscale('log')
            plt.yscale('log')
            
            plt.title("Error scaling of Jacobian and Hessian \n 1st order scaling: "+str(self.fitting_jac)+" \n 2nd order scaling: "+str(self.fitting_hess))
            plt.xlabel('<x-y>')
            plt.ylabel('<f(x-y)> - <J(x)(x-y)> (and -1/2<((x-y),H(x)(x-y))>)')
            
            return fig
            
        else: print("first execute do_test")

    def get_plot_data(self):
        
        xplot = self.epsrange[(self.epsrange>=self.epsplotrange.min()) & (self.epsrange<=self.epsplotrange.max())]
        yplot_jac = self.error_jac[(self.epsrange>=self.epsplotrange.min()) & (self.epsrange<=self.epsplotrange.max())]
        yplot_hess = self.error_hess[(self.epsrange>=self.epsplotrange.min()) & (self.epsrange<=self.epsplotrange.max())]
        
        return xplot, yplot_jac, yplot_hess
    
    
class test_bilinear():
    import numpy as np
    import matplotlib.pyplot as plt
    
    """
    Test errorscaling of jacobian and bilinear for rk4 
    """
    def __init__(self,tend,jac,dim,hessian = None ,para=None,n=100,time=0,epsrange = 10**np.arange(-4,0,0.5),tendtimeord=0,jactimeord=0,hessianconst = True):
        
        tendrk4 , jacrk4, self.rk4_jac_tend = rk4_constructor(jac,tend, dim)
        if hessianconst and callable(hessian):
            _,bilinear = rk4_bilinear_constructor(hessian(0,np.random.rand(dim),para),jac,tend,dim)
        elif hessianconst and not callable(hessian):
            _,bilinear = rk4_bilinear_constructor(hessian,jac,tend,dim)
        elif not hessianconst and not callable(hessian): raise ValueError("Hessian should be not constant, but is not callable.")
        
        
        self.hessianconst = hessianconst
        self.hessian = lambda t,x,dt : bilinear(x,t,dt,para)
        
        self.tend = lambda t,x,dt: tendrk4(x,t,dt,para)
        self.jac = lambda t,x,dt: jacrk4(x,t,dt,para)
        
        self.dim = dim
        self.n = n
        self.time = time
        self.epsrange = epsrange
        self.tendtimeord = tendtimeord
        self.jactimeord = jactimeord
        self.testdone = False
        self.scalar = lambda x,y : np.sum(np.dot(x.reshape(1,-1),y.reshape(-1,1)), dtype = np.float64 )
        self.norm = lambda x : np.sqrt(self.scalar(x,x), dtype = np.float64)
        self.normalize = lambda x : x/self.norm(x)
        self.corr = lambda x,y: self.scalar(self.normalize(x),self.normalize(y))
        
    def do_test(self, scale = 1, T = 0.001, dt = 0.001, rescale = False,Ihessian = None,Vhessian = None, sparse = False):
        
        """
        """
        if sparse:
            if not isinstance(Ihessian,type(None)) and not isinstance(Vhessian,type(None)):
                self.Ihessian = Ihessian
                self.Vhessian = Vhessian
            else:
                if self.jactimeord==0: 
                    self.Ihessian,self.Vhessian = sparse_detect(self.hessian(self.time,np.random.rand(self.dim),dt))
                else:
                    self.Ihessian,self.Vhessian = sparse_detect(self.hessian(np.random.rand(self.dim),self.time,dt))
        
        nsteps = int(np.floor(T/dt))
        
        tend = self.tend
        self.correlation_jac = np.zeros((len(self.epsrange)),dtype = np.float64)
        self.error_jac = np.zeros((len(self.epsrange)),dtype = np.float64)
        self.correlation_hess = np.zeros((len(self.epsrange)),dtype = np.float64)
        self.error_hess = np.zeros((len(self.epsrange)),dtype = np.float64)
        
        self.jacdiff = np.zeros((len(self.epsrange)),dtype = np.float64)
        self.hessdiff= np.zeros((len(self.epsrange)),dtype = np.float64)
        
        for ieps,eps in enumerate(self.epsrange):
            print(ieps,eps)
            for i in range(0, self.n):
                testa = np.float64(self.normalize(np.random.rand(self.dim)))*scale
                testb = np.float64(testa+eps*self.normalize(np.random.rand(self.dim)))
                hessdiff = np.zeros(self.dim)
                jacdiff = np.zeros(self.dim)
                vj = testb-testa
                vh = testb-testa
                
                for nstep in range(0,nsteps):
                    print(eps,nstep,nsteps)
                    
                    if self.tendtimeord==0:
                        tendb = tend(self.time,testb,dt)
                        tenda = tend(self.time,testa,dt)
                    else:
                        tendb = tend(testb,self.time,dt)
                        tenda = tend(testa,self.time,dt)
                    
                    
                    if self.jactimeord==0: jacdiff = np.matmul(self.jac(self.time,testa,dt),vj)
                    else:             jacdiff = np.matmul(self.jac(testa,self.time,dt),vj)

                    if self.jactimeord==0: jacdiffsteph = np.matmul(self.jac(self.time,testa,dt),vh)
                    else:             jacdiffsteph = np.matmul(self.jac(testa,self.time,dt),vh)
                    
                    if sparse:
                        hessdiff = (jacdiffsteph
                                    + np.float64(1/2)*hessian_biproduct_sparse_vec_notconst(self.Ihessian,self.hessian(self.time,testa,dt),self.dim, vh, vh))
                    else:
                        if self.jactimeord==0: 
                            hessdiff = (jacdiffsteph 
                                        + np.float64(1/2)*np.einsum('...ij,i,j',self.hessian(self.time,testa,dt),vh,vh , dtype=np.float64))
                        else:
                            hessdiff = (jacdiffsteph 
                                         + np.float64(1/2)*np.einsum('...ij,i,j',self.hessian(testa,self.time,dt),vh,vh , dtype=np.float64))
                    
                    testa = testa + tenda
                    testb = testb + tendb
                    
                    vj = vj + jacdiff
                    vh = vh + hessdiff
                    
                    
                diff_nonlin = (testb - testa)
                    
                
                self.correlation_jac[ieps]  = self.correlation_jac[ieps] + np.corrcoef(diff_nonlin,vj)[0,1]
                self.error_jac[ieps]        = self.error_jac[ieps] + self.norm(diff_nonlin-vj)
                
                self.correlation_hess[ieps] = self.correlation_hess[ieps] + np.corrcoef(diff_nonlin,vh)[0,1]
                self.error_hess[ieps]       = self.error_hess[ieps] + self.norm(diff_nonlin-vh)
                
                self.jacdiff[ieps]          = self.jacdiff[ieps] + self.norm(vj)
                self.hessdiff[ieps]         = self.hessdiff[ieps] + self.norm(vh)
                
            self.error_jac[ieps]        = self.error_jac[ieps]          /np.float64(self.n)
            self.correlation_hess[ieps] = self.correlation_hess[ieps]   /np.float64(self.n)
            self.error_hess[ieps]       = self.error_hess[ieps]         /np.float64(self.n)
            self.jacdiff[ieps]          = self.jacdiff[ieps]            /np.float64(self.n)
            self.hessdiff[ieps]         = self.hessdiff[ieps]           /np.float64(self.n)
        self.testdone = True
    
    def show_scaling(self, epsfitrange = None, epsplotrange = None, deg = 1):
        if isinstance(epsfitrange, type(None)): epsfitrange = np.array([self.epsrange.min(), self.epsrange.max()])
        if not isinstance(epsfitrange,np.ndarray): 
            epsfitrange = np.array(epsfitrange)
        self.epsfitrange = epsfitrange
        if isinstance(epsplotrange, type(None)): epsplotrange = np.array([self.epsrange.min(), self.epsrange.max()])
        if not isinstance(epsplotrange,np.ndarray): 
            epsplotrange = np.array(epsplotrange)
        self.epsplotrange = epsplotrange
        if self.testdone:
            x = self.epsrange[(self.epsrange>=epsfitrange.min()) & (self.epsrange<=epsfitrange.max())]
            y = self.error_jac[(self.epsrange>=epsfitrange.min()) & (self.epsrange<=epsfitrange.max())]
            
            fig = plt.figure()
            self.logx = np.log(np.float64(x))
            self.logy = np.log(np.float64(y))
            self.coeffs = np.polyfit(self.logx,self.logy,deg=deg,full = True)
            poly = np.poly1d(self.coeffs[0])
            yfit = lambda x: np.exp(poly(self.logx))
            
            xplot = self.epsrange[(self.epsrange>=epsplotrange.min()) & (self.epsrange<=epsplotrange.max())]
            yplot = self.error_jac[(self.epsrange>=epsplotrange.min()) & (self.epsrange<=epsplotrange.max())]
            
            
            plt.plot(xplot,yplot,linestyle='None',marker='o',markerfacecolor = 'b')
            plt.plot(np.float64(x),yfit(np.float64(x)),color ='k',linestyle=':')
            
            fitting_jac = np.round(self.coeffs[0][0],2)
            
            x = self.epsrange[(self.epsrange>=epsfitrange.min()) & (self.epsrange<=epsfitrange.max())]
            y = self.error_hess[(self.epsrange>=epsfitrange.min()) & (self.epsrange<=epsfitrange.max())]
            
            self.logx = np.log(np.float64(x))
            self.logy = np.log(np.float64(y))
            self.coeffs = np.polyfit(self.logx,self.logy,deg=deg,full = True)
            poly = np.poly1d(self.coeffs[0])
            yfit = lambda x: np.exp(poly(self.logx))
            
            fitting_hess = np.round(self.coeffs[0][0],2)
            
            xplot = self.epsrange[(self.epsrange>=epsplotrange.min()) & (self.epsrange<=epsplotrange.max())]
            yplot = self.error_hess[(self.epsrange>=epsplotrange.min()) & (self.epsrange<=epsplotrange.max())]
            
            plt.plot(xplot,yplot,linestyle='None',marker='o',markerfacecolor = 'k')
            plt.plot(x,yfit(x),color ='k',linestyle=':')
            plt.xscale('log')
            plt.yscale('log')
            plt.title("Error scaling of Jacobian and Hessian \n 1st order scaling: "+str(fitting_jac)+" \n 2nd order scaling: "+str(fitting_hess))
            plt.xlabel('<x-y>')
            plt.ylabel('<f(x-y)> - <J(x)(x-y)> (and -1/2<((x-y),H(x)(x-y))>)')
            
            return fig
            
        else: print("first execute do_test")

    def get_plot_data(self):
        
        
        xplot = self.epsrange[(self.epsrange>=self.epsplotrange.min()) & (self.epsrange<=self.epsplotrange.max())]
        yplot_jac = self.error_jac[(self.epsrange>=self.epsplotrange.min()) & (self.epsrange<=self.epsplotrange.max())]
        yplot_hess = self.error_hess[(self.epsrange>=self.epsplotrange.min()) & (self.epsrange<=self.epsplotrange.max())]
        
        return xplot, yplot_jac, yplot_hess