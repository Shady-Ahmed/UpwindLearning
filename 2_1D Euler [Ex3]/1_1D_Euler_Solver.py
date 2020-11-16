# -*- coding: utf-8 -*-
"""
Data generation for 1D Shock-tube [Sod's problem]
This correpsonds to Example 3 for the following paper:
    "Interface learning of multiphysics and multiscale systems",
     Physical Review E, 2020
     
For questions, comments, or suggestions, please contact Shady Ahmed,
PhD candidate, School of Mechanical and Aerospace Engineering, 
Oklahoma State University. @ shady.ahmed@okstate.edu
last checked: 11/16/2020
"""

#%% Import libraries
import numpy as np
import matplotlib.pyplot as plt

import os
import sys

#%% Define Functions

## Initial condition [problem definition]
def init(nx,dx):
    
    lx = dx*nx
    x = np.linspace(-3*dx,lx+3*dx,nx+7)
    q = np.zeros([nx+7,3]) #3 ghost points in each direction
        
    # define initial conditions [Sod's problem]
    rhoL = 1.0
    uL = 0.0
    pL = 1.0
    
    rhoR = 0.125
    uR = 0.0
    pR = 0.1
    
    x0 = 0.5
    gamma = 1.4
    tmax = 0.2
    
   
    #construction initial conditions for conserved variables 
    for i in range(nx+7):
        if x[i] <= x0 + 1e-6:
            r = rhoL
            u = uL
            p = pL
        else:
            r = rhoR
            u = uR
            p = pR
	
        e=p/(r*(gamma-1.0))+0.5*u*u
        #conservative variables 
        q[i,0] = r
        q[i,1] = r*u
        q[i,2] = r*e

    return tmax, gamma, q


## compute time step from cfl
def timestep(nx,q,dx,cfl,gamma):
    #Spectral radius of Jacobian
    sm = 0.0
    for i in range(nx+1):
        l1=np.abs(q[i,1]/q[i,0])
        l2=np.abs(q[i,1]/q[i,0] + np.sqrt(gamma*((gamma-1.0)*(q[i,2]-0.5*q[i,1]*q[i,1]/q[i,0]))/q[i,0]))
        l3=np.abs(q[i,1]/q[i,0] - np.sqrt(gamma*((gamma-1.0)*(q[i,2]-0.5*q[i,1]*q[i,1]/q[i,0]))/q[i,0]))
        rad = np.max([l1,l2,l3])
        if rad>sm:
           sm = rad   
         
    dt = cfl * dx / sm

    return dt


## 1st order Euler scheme
def euler1st(nx,dx,dt,q,ib,gamma):
    s = rhs(nx,dx,q,ib,im,gamma)
    q[3:nx+4,:] = q[3:nx+4,:] + dt*s
    return q


## TVD Runge Kutta 3rd order
def tvdrk3(nx,dx,dt,q,ib,gamma):
    
    a = 1.0/3.0
    b = 2.0/3.0
    u = np.zeros([nx+7,3])

    #1st step
    s = rhs(nx,dx,q,ib,gamma)
    u[3:nx+4,:] = q[3:nx+4,:] + dt*s

    #2nd step
    s = rhs(nx,dx,u,ib,gamma)
    u[3:nx+4,:] = 0.75*q[3:nx+4,:] + 0.25*u[3:nx+4,:] + 0.25*dt*s

    #3rd step
    s = rhs(nx,dx,u,ib,gamma)
    q[3:nx+4,:] = a*q[3:nx+4,:] + b*u[3:nx+4,:] + b*dt*s

    return q
   

## Classical Runge Kutta 4th order
def crk4(nx,dx,dt,q,ib,gamma):
    
    a = 1.0/6.0
    u = np.zeros([nx+7,3])

    s1 = rhs(nx,dx,q,ib,im,gamma)
    u[3:nx+4,:] = q[3:nx+4,:] + 0.5*dt*s1
    
    s2 = rhs(nx,dx,u,ib,im,gamma)
    u[3:nx+4,:] = q[3:nx+4,:] + 0.5*dt*s2

    s3 = rhs(nx,dx,u,ib,im,gamma)
    u[3:nx+4,:] = q[3:nx+4,:] + dt*s3

    s4 = rhs(nx,dx,u,ib,im,gamma)
    q[3:nx+4,:] = q[3:nx+4,:] + a*dt*(s1+2*s2+2*s3+s4)

    return q


## Computing Right Hand Side
def rhs(nx,dx,q,ib,gamma):

    # Apply boundary conditions
    if ib ==1: # 1st order non-reflective (transmissive)
        q[2,:]  = q[3,:]
        q[1,:]  = q[3,:]
        q[0,:]  = q[3,:]
        q[nx+4,:] = q[nx+3,:]
        q[nx+5,:] = q[nx+3,:]
        q[nx+6,:] = q[nx+3,:]
    
    elif ib==2: # 2nd order non-reflective (transmissive)
    
        q[2,:]  = 2.0*q[3,:] - q[4,:] 
        q[1,:]  = 2.0*q[2,:] - q[3,:] 
        q[0,:]  = 2.0*q[1,:] - q[2,:] 
        q[nx+4,:] = 2.0*q[nx+3,:] - q[nx+2,:] 
        q[nx+5,:] = 2.0*q[nx+4,:] - q[nx+3,:] 
        q[nx+6,:] = 2.0*q[nx+5,:] - q[nx+4,:] 
    
    else: # 3rd order non-reflective (transmissive)
    
        q[2,:]  = 3.0*q[3,:] - 3.0*q[4,:] + q[5,:] 
        q[1,:]  = 3.0*q[2,:] - 3.0*q[3,:] + q[4,:]
        q[0,:]  = 3.0*q[1,:] - 3.0*q[2,:] + q[3,:] 
        q[nx+4,:] = 3.0*q[nx+3,:] - 3.0*q[nx+2,:] + q[nx+1,:] 
        q[nx+5,:] = 3.0*q[nx+4,:] - 3.0*q[nx+3,:] + q[nx+2,:] 
        q[nx+6,:] = 3.0*q[nx+5,:] - 3.0*q[nx+4,:] + q[nx+3,:] 
    	
        
    # Reconstruction scheme
    qL,qR = weno5(nx,q)

    fL = flux(nx,qL,gamma)
    fR = flux(nx,qR,gamma)

    # Spectral radius of Jacobian
    rad = np.zeros(nx+5)
    for i in range(2,nx+5):
        l1 = np.abs(q[i,1]/q[i,0])
        l2 = np.abs(q[i,1]/q[i,0] + np.sqrt(gamma*((gamma-1.0)*(q[i,2]-0.5*q[i,1]*q[i,1]/q[i,0]))/q[i,0]))
        l3 = np.abs(q[i,1]/q[i,0] - np.sqrt(gamma*((gamma-1.0)*(q[i,2]-0.5*q[i,1]*q[i,1]/q[i,0]))/q[i,0]))
        rad[i] = np.max([l1,l2,l3])
    
    # Propagation speed
    ps = np.zeros(nx+4)
    for i in range(2,nx+4):
        ps[i] = np.max([rad[i],rad[i+1]])

    # Compute fluxes with Rusanov
    f = np.zeros([nx+4,3])
    for i in range(2,nx+4):
    	f[i,:] = 0.5*((fR[i,:]+fL[i,:]) - ps[i]*(qR[i,:]-qL[i,:]))

    # Compute rhs
    s = np.zeros([nx+4,3])
    for i in range(3,nx+4):
    	s[i,:] = -(f[i,:]-f[i-1,:])/dx
           
    return s[3:nx+4]



## !5th order WENO
def weno5(nx,q):
    qL = np.zeros([nx+4,3])
    qR = np.zeros([nx+4,3])
    
    eps = 1.0e-6
    h = 13.0/12.0
    g = 1.0/6.0
    
    a = 3.0/10.0
    b = 3.0/5.0
    c = 1.0/10.0
    
    for i in range(2,nx+4):
  
        b0 = h*(q[i,:]-2.0*q[i+1,:]+q[i+2,:])**2 \
           + 0.25*(3.0*q[i,:]-4.0*q[i+1,:]+q[i+2,:])**2
        b1 = h*(q[i-1,:]-2.0*q[i,:]+q[i+1,:])**2 \
           + 0.25*(q[i-1,:]-q[i+1,:])**2
        b2 = h*(q[i-2,:]-2.0*q[i-1,:]+q[i,:])**2 \
           + 0.25*(q[i-2,:]-4.0*q[i-1,:]+3.0*q[i,:])**2
    
        a0 = a/(eps+b0)**2
        a1 = b/(eps+b1)**2
        a2 = c/(eps+b2)**2
        
        w0 = a0/(a0+a1+a2)
        w1 = a1/(a0+a1+a2)
        w2 = a2/(a0+a1+a2)
        
        qL[i,:]=g*w0*(-q[i+2,:]+5.0*q[i+1,:]+2.0*q[i,:]) \
               +g*w1*(-q[i-1,:]+5.0*q[i,:]+2.0*q[i+1,:]) \
               +g*w2*(2.0*q[i-2,:]-7.0*q[i-1,:]+11.0*q[i,:])
        
        a0 = c/(eps+b0)**2
        a1 = b/(eps+b1)**2
        a2 = a/(eps+b2)**2
        
        w0 = a0/(a0+a1+a2)
        w1 = a1/(a0+a1+a2)
        w2 = a2/(a0+a1+a2)
        
        qR[i,:]=g*w0*(2.0*q[i+3,:]-7.0*q[i+2,:]+11.0*q[i+1,:])\
               +g*w1*(-q[i+2,:]+5.0*q[i+1,:]+2.0*q[i,:]) \
               +g*w2*(-q[i-1,:]+5.0*q[i,:]+2.0*q[i+1,:]) 
    
    return qL,qR 



## Computing fluxes from conserved quantities
def flux(nx,q,gamma):
    f = np.zeros([nx+4,3])
    for i in range(2,nx+4):
        f[i,0] = q[i,1]
        f[i,1] = q[i,1]*q[i,1]/q[i,0] + (gamma-1.0)*(q[i,2]-0.5*q[i,1]*q[i,1]/q[i,0])
        f[i,2] = q[i,1]*q[i,2]/q[i,0] + (gamma-1.0)*q[i,1]/q[i,0]*(q[i,2]-0.5*q[i,1]*q[i,1]/q[i,0])

    return f


## Computing results
# Compute primitive variables from conservative variables
def outresult(fname,nx,dx,time,q):
    
    lx = dx*nx
    x = np.linspace(-3*dx,lx+3*dx,nx+7)

    #computing results at cell centers:
    r = np.zeros(nx+4)
    u = np.zeros(nx+4)
    e = np.zeros(nx+4)
    p = np.zeros(nx+4)
    h = np.zeros(nx+4)
    a = np.zeros(nx+4)
    m = np.zeros(nx+4)
    for i in range(3,nx+4):
        r[i]= q[i,0]                        #rho: density 
        u[i]= q[i,1]/q[i,0]                 #u: velocity
        e[i]= q[i,2]/q[i,0]                 #e: internal energy
        p[i]= (gamma-1.0)*(q[i,2]-0.5*q[i,1]*q[i,1]/q[i,0]) #p: pressure
        h[i]= e[i] + p[i]/r[i]              #h: static enthalpy
        a[i]= np.sqrt(gamma*p[i]/r[i])      #a: speed of sound
        m[i]= u[i]/a[i]                     #m: Mach number
    
    #writing results at cell centers:
    np.savez(fname, time=time, x=x[3:nx+4], r=r[3:nx+4],\
                          u=u[3:nx+4], p=p[3:nx+4], e=e[3:nx+4],\
                          h=h[3:nx+4], a=a[3:nx+4], m=m[3:nx+4])

    return 


#%% Inputs

nx = 400        #nx;
cfl = 0.5   	#cfl;
it = 1      	#it;[0]Euler1st,[1]RK3,[2]RK4
ib = 1      	#ib;[1]1st,[2]2nd,[3]3rd
nt = 90000000   #nt;maximum number of time step

lx = 1

dx = lx/nx

#%%

# create folder
if os.path.isdir("./Data"):
    print('Data folder already exists')
else: 
    print('Creating Data folder')
    os.makedirs("./Data")
        
        
# Initial condition
tmax, gamma, q = init(nx,dx)

iend = 0    
time = 0.0

fname = 'Data/cell_k=0.npz'
#writing results
outresult(fname,nx,dx,time,q)


#Time integration
dtmin = 1e-4
for k in range(1,nt+1):

    #compute time step from cfl
    dt = timestep(nx,q,dx,cfl,gamma)
    if dt<dtmin:
        dtmin = dt

    dt = dtmin
    #check for final time step
    if (time+dt >=tmax) :
        dt = tmax-time
        iend=1
    
    time = time + dt
    if it == 0: # 1st order Euler
        q = euler1st(nx,dx,dt,q,ib,gamma)
    elif it ==1:  #Third-order RK
        q = tvdrk3(nx,dx,dt,q,ib,gamma)
    else: #4th order RK
        q = crk4(nx,dx,dt,q,ib,gamma)
    
    print(k,time,dt)
    fname = 'Data/cell_k='+str(k)+'.npz'
    #writing results
    outresult(fname,nx,dx,time,q)
    if iend == 1:
        break 

outresult('Data/final.npz',nx,dx,time,q)
