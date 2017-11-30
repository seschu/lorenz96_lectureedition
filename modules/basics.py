#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

root = "/scratch/uni/u234/u234069/lorenz96_lectureedition"

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
            

def rk4_jacobian(jac,f,dim):

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
        
        jac1res = jac1(t,y,p,dt)
        jac2res = jac2(t,y,p,dt)
        jac3res = jac3(t,y,p,dt)
        jac4res = jac4(t,y,p,dt)
        
        deriv_k2 = jac2res + np.matmul(jac2res,dt/2*jac1res )
        deriv_k3 = jac3res + np.matmul(jac3res,dt/2*deriv_k2)
        deriv_k4 = jac4res + np.matmul(jac4res,dt  *deriv_k3) 
        return dt/6*(jac1res + 2* deriv_k2 + 2* deriv_k3 + deriv_k4)
    
    def all_rk4(V,y,t,dt,p):
        
        k1res = f(t,y,p)
        k2res = f(t,y + dt/2*k1res,p)
        k3res = f(t,y + dt/2*k2res,p)
        k4res = f(t,y + dt*k3res,p)
        
        jac1res = jac1(t,y,p,dt)
        jac2res = jac2(t,y,p,dt,k1res)
        jac3res = jac3(t,y,p,dt,k2res)
        jac4res = jac4(t,y,p,dt,k3res)
        
        deriv_k2 = jac2res + np.matmul(jac2res,dt/2*jac1res )
        deriv_k3 = jac3res + np.matmul(jac3res,dt/2*deriv_k2)
        deriv_k4 = jac4res + np.matmul(jac4res,dt  *deriv_k3)
        
        return V + dt/6*np.matmul((jac1res + 2* deriv_k2 + 2* deriv_k3 + deriv_k4), V), y + dt/6*(k1res + 2 * k2res + 2* k3res + k4res)
    
    return func_rk4, jac_rk4, all_rk4