#
# first lecture - Data Assimilation
# conceptual climate model
#

from lorenz96 import *


######################
# Start Experiments  #
######################   
 
# run two experiments with different F
    
for F in [10, 20]:    
    
    ######################
    # Setting Parameters #
    ######################   
    print(niceprint("Setting Parameters",level = 0))
    
    # p is a dictionary
    p={'dimX' : 36,  # also called K
       'dimY' : 10,  # also called J
          'h' : 1,   # coupling strength
          'c' : 10,  # time scale 
          'b' : 10,  # length scale 
          'F' : F}   # forcing
    print(niceprint(str(p).replace("{","").replace("}","").replace("'","").replace(","," \\n "),level = 2, width = 15))
    
        
    #############################################
    # stationary state plus random perturbation #
    #############################################
    print(niceprint("Stationary State plus Random Perturbation ",level = 0))
    
    x0 = L96_stationary_state(p) # initial state (equilibrium)
    x0 = x0 + 0.1*np.hstack([np.random.rand(p['dimX'])*10,np.random.rand(p['dimX']*p['dimY'])]) # add small perturbation to 20th variable
    
        
    ######################
    #      Spin Up       #
    ######################
    print(niceprint("Spin Up",level = 0))
    
    t_spinup = np.arange(0.0, 100.0, 0.1)
    
    # setup time and result matrix
    print(niceprint("setup time and result matrix",level = 2))
    
    time = t_spinup
    
    # initialize integrator class
    print(niceprint("initialize integrator class",level = 2))
    r = ode(L96_tendency, L96_jacobian).set_integrator('lsoda', with_jacobian=True, rtol=10E-8, atol=10E-8, nsteps = 10**5)
    r.set_initial_value(x0, time[0]).set_f_params(p).set_jac_params(p)
    
    # do integration in time
    print(niceprint("do integration in time",level = 2))
    for nstep ,( told , tnew ) in enumerate(zip(time[0:-1],time[1:])):
        r.integrate(r.t+tnew-told)
    
    # save initial state
    x0=r.y
                    
    ######################
    #      Main Run      #  
    ######################
    print(niceprint("Main Run",level = 0))

    time = np.arange(0.0, 100, 0.1)
    x = np.zeros((len(time), p['dimX']*p['dimY']+p['dimX']))
    
    # initialize integrator class
    print(niceprint("initialize integrator class",level = 2))
    
    r = ode(L96_tendency, L96_jacobian).set_integrator('lsoda', with_jacobian=True, rtol=10E-8, atol=10E-8, nsteps = 10**5)
    r.set_initial_value(x0, time[0]).set_f_params(p).set_jac_params(p)
    
    # do integration in time
    print(niceprint("do integration in time",level = 2))
    for nstep ,( told , tnew ) in enumerate(zip(time[0:-1],time[1:])):
        x[nstep,:]=r.y
        r.integrate(r.t+tnew-told)
    x[-1,:]=r.y    
    
    
    ######################
    #     Plotting       #  
    ######################
    print(niceprint("Plotting",level = 0))

    import matplotlib.pyplot as plt
    
    ## plot distribution of X modes
    
    fig=plt.figure()
    plt.hist(x[:,:p['dimX']].flatten(),normed=1,bins=50)
    plt.ylabel('X')
    plt.xlabel('pdf')
    plt.title('PDF of X modes')
    fig.savefig("pdf_X_modes_F_"+str(p['F'])+".pdf")
    plt.close(fig)
    
    ## plot hovmoeller diagram of X modes
    
    fig=plt.figure()
    X, Y = np.meshgrid(range(1,p['dimX']+1),time[0:15*10])
    plt.contourf(X,Y,x[0:15*10,:p['dimX']])
    plt.ylabel('time')
    plt.xlabel('X modes')
    plt.title('Hovmüller diagramm of X modes')
    fig.savefig("Hovmöller_X_modes_F_"+str(p['F'])+".pdf")
    plt.close(fig)
    
    
    ## plot distribution of Y modes
    
    fig=plt.figure()
    plt.hist(x[:,p['dimX']:].flatten(),normed=1,bins=50)
    plt.ylabel('Y')
    plt.xlabel('pdf')
    plt.title('PDF of Y modes')
    fig.savefig("pdf_Y_modes_F_"+str(p['F'])+".pdf")
    plt.close(fig)
    
    ## plot hovmoeller diagram of Y modes
    
    fig=plt.figure()
    X, Y = np.meshgrid(range(1,p['dimY']+1),time[0:15*10])
    plt.contourf(X,Y,x[0:15*10,p['dimX']:p['dimX']+p['dimY']])
    plt.ylabel('time')
    plt.xlabel('Y modes')
    plt.title('Hovmüller diagramm of Y modes')
    fig.savefig("Hovmöller_Y_modes_F_"+str(p['F'])+".pdf")
    plt.close(fig)
    
    ## plot temporal autocorrelation in time of Y modes
    
    fig=plt.figure()
    ac = np.zeros(100)
    for i in range(p['dimX'],p['dimX']+p['dimX']*p['dimY']):
        ts = pd.Series(x[:,i])
        ac = ac + np.array(list(map(ts.autocorr,range(0,100))))/(p['dimX']*p['dimY'])
    plt.plot(time[0:100], ac)
    plt.ylabel('acf')
    plt.xlabel('time lag')
    plt.title('Autocorrelation of Y modes')
    fig.savefig("Autocorrelation_time_Y_modes_F_"+str(p['F'])+".pdf")
    plt.close(fig)
    
    ## plot temporal autocorrelation in time of X modes
    
    fig=plt.figure()
    ac = np.zeros(100)
    for i in range(0,p['dimX']):
        ts = pd.Series(x[:,i])
        ac = ac + np.array(list(map(ts.autocorr,range(0,100))))/p['dimX']
    plt.plot(time[0:100], ac)
    plt.ylabel('acf')
    plt.xlabel('time lag')
    plt.title('Temporal Autocorrelation of X modes')
    fig.savefig("Autocorrelation_time_X_modes_F_"+str(p['F'])+".pdf")
    plt.close(fig)
    
    ## plot spatial autocorrelation in time of Y modes
    
    fig=plt.figure()
    ac = np.zeros(int(p['dimY']*2))
    for i,t in enumerate(time):
        ts = pd.Series(x[i,p['dimX']:])
        ac = ac + np.array(list(map(ts.autocorr,range(0,p['dimY']*2))))/len(time)
    plt.plot(range(0,p['dimY']*2), ac)
    plt.ylabel('acf')
    plt.xlabel('time lag')
    plt.title('Autocorrelation of Y modes')
    fig.savefig("Autocorrelation_spatial_Y_modes_F_"+str(p['F'])+".pdf")
    plt.close(fig)
    
    ## plot spatial autocorrelation in time of X modes
    
    fig=plt.figure()
    ac = np.zeros(int(p['dimX']/2))
    for i,t in enumerate(time):
        ts = pd.Series(x[i,:p['dimX']])
        ac = ac + np.array(list(map(ts.autocorr,range(0,int(p['dimX']/2)))))/len(time)
    plt.plot(range(0,int(p['dimX']/2)), ac)
    plt.ylabel('acf')
    plt.xlabel('time lag')
    plt.title('Autocorrelation of X modes')
    fig.savefig("Autocorrelation_spatial_X_modes_F_"+str(p['F'])+".pdf")
    plt.close(fig)

    ## plot X mode vs Y mode forcing
    
    fig=plt.figure()
    for i in range(0,p['dimX']):
        plt.plot(x[:,i],p['h']*p['c']/p['b']*np.sum(x[:,p['dimY']*i+p['dimX']:(i+1)*p['dimY']+p['dimX']], axis = 1) , marker ='.' , linestyle='None',color='k')
    plt.xlabel('X_k')
    plt.ylabel(r'$\frac{hc}{b}\sum_{i=J(k-1)+1}^{kJ} Y_j$')
    plt.title('X modes vs forcing of Y modes')
    fig.savefig("Forcing_YonX_modes_F_"+str(p['F'])+".pdf")
    plt.close(fig)
    
