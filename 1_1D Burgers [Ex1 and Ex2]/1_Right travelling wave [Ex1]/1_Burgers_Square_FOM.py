# -*- coding: utf-8 -*-
"""
Data generation for 1D Burgers problem with an initial condition of square wave
This correpsonds to Example 1 for the following paper:
    "Interface learning of multiphysics and multiscale systems",
     Physical Review E, 2020
     
For questions, comments, or suggestions, please contact Shady Ahmed,
PhD candidate, School of Mechanical and Aerospace Engineering, 
Oklahoma State University. @ shady.ahmed@okstate.edu
last checked: 11/10/2020
"""

#%% Import libraries
import numpy as np
import matplotlib.pyplot as plt

import os
import sys
#%% Define Functions
#-----------------------------------------------------------------------------!
#compute rhs for numerical solutions
#  r = -u*u' + nu*u''
#-----------------------------------------------------------------------------!
def rhs(nx,dx,nu1,nu2,g1,g2,nxb,u):
    r = np.zeros(nx+1)
    gg = np.zeros(nx+1)
    nuu = np.zeros(nx+1)
    gg[0:nxb+1] = g1
    gg[nxb+1:] = g2
    nuu[0:nxb+1] = nu1
    nuu[nxb+1:] = nu2

    r[1:nx] = (1/(dx*dx)) \
             *nuu[1:nx]*(u[2:nx+1] - 2*u[1:nx] + u[0:nx-1]) \
             - gg[1:nx]*u[1:nx]\
             - (1.0/3.0)*(u[2:nx+1]+u[0:nx-1]+u[1:nx])*(u[2:nx+1]-u[0:nx-1])/(2.0*dx)         
    return r

#%% Main program:
    
# Inputs
nx =  4*1024  #spatial resolution
lx = 1.0    #spatial domain
dx = lx/nx
x = np.linspace(0, lx, nx+1)

nu1 = 1e-2   #control dissipation
nu2 = 1e-4   #control dissipation
g1 = 0 #friction
g2 = 1 #friction

tm = 1      #maximum time
dt = 2.5e-6   #solver timestep
nt = round(tm/dt)
t = np.linspace(0, tm, nt+1)

ns = 4000 #number of snapshots to save
freq = round(nt/ns)


#%%

for ii in range(1,8):
    nxb= int(ii*nx/8)
    uFOM = np.zeros((nx+1,ns+1))
    
    #define initial conditions [square wave]
    uu = np.zeros(nx+1)
    for i in range(nx+1):
        if(i <= nxb):
            uu[i] = 1.0
        else:
            uu[i] = 0.0
    
    # boundary conditions: b.c. (not updated)
    u1 = np.zeros(nx+1)
    uu[0] = 0.0
    uu[nx] = 0.0
    u1[0] = 0.0
    u1[nx]= 0.0       
        
    #check for stability
    neu = np.max([nu1,nu2])*dt/(dx*dx)
    cfl = np.max(uu)*dt/dx
    if (neu >= 0.5):
        print('Neu condition: reduce dt')
        sys.exit()
    if (cfl >=  1.0):
        print('CFL condition: reduce dt')
        sys.exit()
          
    #time integration
    if1 = 0
    uFOM[:,0] = uu 
    for jj in range(1,nt+1):
        #RK3 scheme
        # first step        
        rr = rhs(nx,dx,nu1,nu2,g1,g2,nxb,uu)
        u1[1:nx] = uu[1:nx] + dt*rr[1:nx]
        
        # second step
        rr = rhs(nx,dx,nu1,nu2,g1,g2,nxb,u1)
        u1[1:nx] = 0.75*uu[1:nx] + 0.25*u1[1:nx] + 0.25*dt*rr[1:nx]
        	
        # third step
        rr = rhs(nx,dx,nu1,nu2,g1,g2,nxb,u1)
        uu[1:nx] = 1.0/3.0*uu[1:nx] + 2.0/3.0*u1[1:nx] + 2.0/3.0*dt*rr[1:nx]
                     
        # compute and plot velocity field
        if(np.mod(jj,freq) == 0):
            if1=if1+1
            uFOM[:,if1] = uu 
            print([x[nxb],if1])
    
        #compute CFL
        cfl = np.max(uu)*dt/dx
        #check for numerical stability
        if (cfl > 1.0):
            print('Error: CFL limit exceeded')
            break
        
    #%% Saving data
    
    #create data folder
    if os.path.isdir("./Data"):
        print('Data folder already exists')
    else: 
        print('Creating data folder')
        os.makedirs("./Data")
     
    print('Saving data')      
    np.save('./Data/uFOM_xb='+str(nxb/nx)+'_.npy',uFOM)
    
    
    # create plot folder
    if os.path.isdir("./Plots"):
        print('Plots folder already exists')
    else: 
        print('Creating Plots folder')
        os.makedirs("./Plots")
        
    plt.figure(figsize=(8,6))
    plt.plot(x,uFOM[:,::500])
    plt.xlabel('x')
    plt.ylabel('u')
    plt.savefig('./Plots/Burgers_Square_xb='+str(nxb/nx)+'_.png', dpi = 500, bbox_inches = 'tight')
    plt.show()


