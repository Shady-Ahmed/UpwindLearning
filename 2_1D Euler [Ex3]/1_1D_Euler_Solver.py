# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 08:47:48 2020

@author: Shady
"""


import numpy as np
import os


#%% Define Functions

## Initial condition [problem definition]
def init(ip,nx,dx):
    
    lx = dx*nx
    x = np.linspace(-3*dx,lx+3*dx,nx+7)
    q = np.zeros([nx+7,3]) #3 ghost points in each direction
        
    if ip == 1 :  #Sod's problem
    
        rhoL = 1.0
        uL = 0.0
        pL = 1.0
        
        rhoR = 0.125
        uR = 0.0
        pR = 0.1
        
        x0 = 0.5
        gamma = 1.4
        tmax = 0.2
        
    elif ip == 2: #Lax's problem 
          
        rhoL=0.445
        uL=0.698
        pL=3.528
    
        rhoR = 0.5
        uR = 0.0
        pR = 0.571
        
        x0 = 0.5
        gamma = 1.40
        tmax  = 0.12
                    
    elif ip == 3: #123 problem, see Toro !made of a two rarefaction wave 
                  #!low density flows
        rhoL=1.0
        uL=-2.0
        pL=0.4
        
        rhoR =1.0
        uR = 2.0
        pR = 0.4
        
        x0    = 0.5
        gamma = 1.40
        tmax  = 0.15

    elif ip == 4: #High Mach problem, !Xiao, JCP 195, 2004
        
        rhoL=10.0
        uL=2000.0
        pL=500.0
        
        rhoR=20.0
        uR=0.0
        pR=500.0
        
        x0    = 0.5
        gamma = 1.40
        tmax  = 1.75e-4

    elif ip ==  5: #Mach3 problem, !Xiao, JCP 195, 2004

        rhoL=3.857
        uL=0.92
        pL=10.333
        
        rhoR=1.0
        uR=3.55
        pR=1.0
        
        x0    = 0.5
        gamma = 1.40
        tmax  = 0.09    
    
    elif ip == 6:  #Double shock wave
  
        rhoL=3.0
        uL=100.0
        pL=573.0	
        
        rhoR=3.0
        uR=-100.0
        pR=573.0
        
        x0    = 0.5
        gamma = 1.40
        tmax  = 0.01

   
    elif ip == 7:  #Toro's problem, Left Woodward-Collela
                   #severe rest problem, left rarefaction, a contact and right shock

        rhoL=1.0
        uL=0.0
        pL=1000.0	
        
        rhoR=1.0
        uR=0.0
        pR=0.01
        
        x0    = 0.5
        gamma = 1.40
        tmax  = 0.012
          
    elif ip == 8: #Toro's problem, !moving contact discontinuity
    
        rhoL=1.4
        uL=0.1
        pL=1.0
        
        rhoR=1.0
        uR=0.1
        pR=1.0
        
        x0    = 0.5
        gamma = 1.40
        tmax  = 2.0

    elif ip == 9: #peak density, !J.A. Greenough, W.J. Rider / JCP 196 (2004) 259281
    
        rhoL=0.1261192
        uL=8.9047029
        pL=782.92899
        
        rhoR=6.591492
        uR=2.2654207
        pR=3.1544874
        
        x0    = 0.5
        gamma = 1.40
        tmax  = 0.0039
               
    #construction initial conditions for conserved variables 

    if ip == 10: #Shu-Osher problem
          
        gamma = 1.40
        tmax  = 0.18
    
        for i in range(nx+7):
    
            if (x[i] <= 0.1+ 1e-6) :
                r = 3.857143
                u = 2.629369
                p = 10.333333
            else:
                r = 1.0+0.2*np.sin(50.0*x[i])
                u = 0.0
                p = 1.0
                
            e=p/(r*(gamma-1.0))+0.5*u*u
            #conservative variables 
            q[i,0] = r
            q[i,1] = r*u
            q[i,2] = r*e

    else:  #standard shock tube problems
      
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
        #print(l1,l2,l3)
        if rad>sm:
           sm = rad   
         
    dt = cfl * dx / sm

    return dt


## 1st order Euler scheme
def euler1st(nx,dx,dt,q,ib,ir,im,iss,gamma):
    s = rhs(nx,dx,q,ib,ir,im,iss,gamma)
    q[3:nx+4,:] = q[3:nx+4,:] + dt*s
    return q


## TVD Runge Kutta 3rd order
def tvdrk3(nx,dx,dt,q,ib,ir,im,iss,gamma):
    
    a = 1.0/3.0
    b = 2.0/3.0
    u = np.zeros([nx+7,3])

    #1st step
    s = rhs(nx,dx,q,ib,ir,im,iss,gamma)
    u[3:nx+4,:] = q[3:nx+4,:] + dt*s

    #2nd step
    s = rhs(nx,dx,u,ib,ir,im,iss,gamma)
    u[3:nx+4,:] = 0.75*q[3:nx+4,:] + 0.25*u[3:nx+4,:] + 0.25*dt*s

    #3rd step
    s = rhs(nx,dx,u,ib,ir,im,iss,gamma)
    q[3:nx+4,:] = a*q[3:nx+4,:] + b*u[3:nx+4,:] + b*dt*s

    return q
   


## Classical Runge Kutta 4th order
def crk4(nx,dx,dt,q,ib,ir,im,iss,gamma):
    
    a = 1.0/6.0
    u = np.zeros([nx+7,3])

    s1 = rhs(nx,dx,q,ib,ir,im,iss,gamma)
    u[3:nx+4,:] = q[3:nx+4,:] + 0.5*dt*s1
    
    s2 = rhs(nx,dx,u,ib,ir,im,iss,gamma)
    u[3:nx+4,:] = q[3:nx+4,:] + 0.5*dt*s2

    s3 = rhs(nx,dx,u,ib,ir,im,iss,gamma)
    u[3:nx+4,:] = q[3:nx+4,:] + dt*s3

    s4 = rhs(nx,dx,u,ib,ir,im,iss,gamma)
    q[3:nx+4,:] = q[3:nx+4,:] + a*dt*(s1+2*s2+2*s3+s4)

    return q


## Computing Right Hand Side
def rhs(nx,dx,q,ib,ir,im,iss,gamma):

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
    if ir == 1: # MUSCL construction
    	qL,qR = muscl(nx,q,im,iss)
    elif ir ==2: #WENO-3
    	qL,qR = weno3(nx,q)
    else:  #WENO-5
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



## MUSCL scheme
def muscl(nx,q,im,iss):
    qL = np.zeros([nx+4,3])
    qR = np.zeros([nx+4,3])
    
    # compute flux limiter phi
    phi, ksi = limiter(nx,q,iss)

    if im == 1: #1st-order  
        qL[2:nx+4,:]=q[2:nx+4,:]
        qR[2:nx+4,:]=q[3:nx+5,:]

    elif im == 2: #2nd-order upwind
        kappa = -1.0
        for i in range(2,nx+4):
            qL[i,:] = q[i,:]+0.25*((1.0-kappa)*ksi[i,:]*(q[i,:]-q[i-1,:]) \
                            + (1.0+kappa)*phi[i,:]*(q[i+1,:]-q[i,:]) )
                             
            qR[i,:] = q[i+1,:]-0.25*((1.0+kappa)*ksi[i+1,:]*(q[i+1,:]-q[i,:]) \
                               + (1.0-kappa)*phi[i+1,:]*(q[i+2,:]-q[i+1,:]) )
    
    elif im ==3: #Fromm scheme
        kappa = 0.0
        for i in range(2,nx+4):
            qL[i,:]=q[i,:]+0.25*((1.0-kappa)*ksi[i,:]*(q[i,:]-q[i-1,:]) \
                          + (1.0+kappa)*phi[i,:]*(q[i+1,:]-q[i,:]) )
                         
            qR[i,:]=q[i+1,:]-0.25*((1.0+kappa)*ksi[i+1,:]*(q[i+1,:]-q[i,:]) \
                            + (1.0-kappa)*phi[i+1,:]*(q[i+2,:]-q[i+1,:]) )


    elif im == 4: #3rd-order scheme
        kappa = 1.0/3.0  
        for i in range(2,nx+4):
            qL[i,:]=q[i,:]+0.25*((1.0-kappa)*ksi[i,:]*(q[i,:]-q[i-1,:]) \
                          + (1.0+kappa)*phi[i,:]*(q[i+1,:]-q[i,:]) )
                         
            qR[i,]=q[i+1,:]-0.25*((1.0+kappa)*ksi[i+1,:]*(q[i+1,:]-q[i,:]) \
                           + (1.0-kappa)*phi[i+1,:]*(q[i+2,:]-q[i+1,:]) )

    elif im == 5: #centered scheme
        kappa = 1.0 
        for i in range(2,nx+4):  
            qL[i,:]=q[i,:]+0.25*((1.0-kappa)*ksi[i,:]*(q[i,:]-q[i-1,:]) \
                         + (1.0+kappa)*phi[i,:]*(q[i+1,:]-q[i,:]) )
                             
            qR[i,:]=q[i+1,:]-0.25*((1.0+kappa)*ksi[i+1,:]*(q[i+1,:]-q[i,:]) \
                           + (1.0-kappa)*phi[i+1,:]*(q[i+2,:]-q[i+1,:]) )

    else: #2nd-order Kurganov-Tadmor 
        for i in range(2,nx+4):

            qL[i,:]=q[i,:]+0.5*phi[i,:]*(q[i+1,:]-q[i,:])
            qR[i,:]=q[i+1,:]-0.5*phi[i+1,:]*(q[i+2,:]-q[i+1,:])
    
    return qL, qR


## 3rd order WENO
def weno3(nx,q):
    qL = np.zeros([nx+4,3])
    qR = np.zeros([nx+4,3])
    
    eps = 1.0e-6
    a = 2.0/3.0
    b = 1.0/3.0
    
    for i in range(2,nx+4):
      
        b0 = (q[i+1,:] - q[i,:])**2
        b1 = (q[i,:] - q[i-1,:])**2
        
        a0 = a/(eps+b0)**2
        a1 = b/(eps+b1)**2
        w0 = a0/(a0+a1)
        w1 = a1/(a0+a1)
        
        qL[i,:]=0.5*w0*(q[i,:]+q[i+1,:]) + 0.5*w1*(-q[i-1,:]+3.0*q[i,:]) 
        
        a0 = b/(eps+b0)**2
        a1 = a/(eps+b1)**2
        w0 = a0/(a0+a1)
        w1 = a1/(a0+a1)
        
        qR[i,:]=0.5*w0*(-q[i+2,:]+3.0*q[i+1,:])+ 0.5*w1*(q[i+1,:]+q[i,:])

    return qL,qR


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


## Computing flux limiter at nodes
def limiter(nx,q,iss):
    eps=1.0e-24
    phi = np.zeros([nx+5,3])
    ksi = np.zeros([nx+5,3])
    for i in range(2,nx+5):
        for m in range(3):
            a = q[i,m] - q[i-1,m] 
            b = q[i+1,m] - q[i,m]
               
            if np.abs(b) <= eps: 
                phi[i,m] = lim(iss)
            else:	
                r = a/b	
                phi[i,m] = lim_psi(iss,r)
        
            if np.abs(a) <= eps:
                ksi[i,m] = lim(iss)
            else:	
                r = b/a	
                ksi[i,m] = lim_psi(iss,r)
        
    return phi,ksi

## !Limiter function
def lim_psi(iss,r):
    
    if  iss == 1: #CHARM; 
        psi = np.max([0.0,r*(3.0*r+1.0)/(r+1.0)**2])

    elif iss==2: #HCUS;  
        psi = np.max([0.0,1.5*(r+np.abs(r))/(r+2.0)])

    elif iss==3: #HQUICK;  
        psi = np.max([0.0,2.0*(r+np.abs(r))/(r+3.0)])
    
    elif iss==4: #Koren;  
        psi = np.max([0.0,np.min([2.0*r,(1.0+r)/3.0,2.0])]) 
    
    elif iss==5: #minmod;  
        psi = np.max([0.0,np.min([1.0,r])])
    
    elif iss==6: #Monotonized central;  
        psi = np.max([0.0,np.min([2.0*r,(1.0+r)*0.5,2.0])])
    
    elif iss==7: #Osher;
        psi = np.max([0.0,np.min[r,1.5]])
        
    elif iss==8: #ospre;
        psi = np.max([0.0,1.5*(r*r+r)/(r*r+r+1.0)]) 
    
    elif iss==9: #smart;
        psi = np.max([0.0,np.min([2.0*r,0.25+0.75*r,4.0])])
    
    elif iss==10: #Superbee; 
        psi = np.max([0.0,np.min([2.0*r,1.0]),np.min([r,2.0])])
    
    elif iss==11: #Sweby;
        psi = np.max([0.0,np.min([1.5*r,1.0]),np.min([r,1.5])])
    
    elif iss==12: #UMIST;
        psi = np.max([0.0,np.min([2.0*r,0.25+0.75*r,0.75+0.25*r,2.0])])
    
    elif iss==13: #Van Albada-1; 
        psi = np.max([0.0,(r*r + r) / (r*r + 1.0)])
    
    elif iss==14: #Van Albada-2; 
        psi = np.max([0.0,(2.0*r) / (r*r + 1.0)])
    
    elif iss==15: #Van Leer-1;
        psi = np.max([0.0,(r + np.abs(r)) / (1.0 + np.abs(r))])
    
    elif iss==16: #Venkatakrishnan;
        psi = np.max([0.0,(r*r+2.0*r)/(2.0+r+r*r)])
    
    else: #modified minmod;
        psi = np.max([0.0,np.min([1.0,4*r])])  
    
    return psi


## Limiter limits as r goes infinity
def lim(iss):

    if iss == 1: #CHARM; 
        l = 3.0 
    
    elif iss == 2: #HCUS;  
        l = 3.0
    
    elif iss==3: #HQUICK;  
        l = 4.0
    
    elif iss==4: #Koren;  
        l = 2.0
    
    elif iss==5: #minmod; 
        l = 1.0
    
    elif iss==6: #MC; 
        l = 2.0
    
    elif iss==7: #Osher;
        l = 1.5
    
    elif iss==8: #ospre; 
        l = 1.5
    
    elif iss==9: #smart; 
        l = 4.0
    
    elif iss==10: #superbee; 
        l = 2.0
    
    elif iss==11: #Sweby; 
        l = 1.5
    
    elif iss==12: #UMIST; 
        l = 2.0
    
    elif iss==13: #van Albada-1; 
        l = 1.0
    
    elif iss==14: #van Albada-2; 
        l = 0.0
    
    elif iss==15: #van Leer-1; 
        l = 2.0
    
    elif iss==16: #Venkatakrishnan;
        l = 1.0
      
    else: #modified minmod; 
        l = 1.0
    
    return l


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
ip = 1          #ip;[1]Sod,[2]Lax,[3]123,[4]HM,[5]M3,[6]2S,[7]L,[8]MC,[9]peak,[10]SO
iss = 5         #iss;[1]CHARM,[5]minmod,[6]MC,[8]ospre,[10]superbee,[15]vanLeer
it = 1      	#it;[0]Euler1st,[1]RK3,[2]RK4
ir = 3      	#ir;[1]MUSCL,[2]WENO3,[3]WENO5
im = 1      	#im;[1]1st,[2]up,[3]Fromm,[4]3rd,[5]cen,[6]KT
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
tmax, gamma, q = init(ip,nx,dx)

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
        q = euler1st(nx,dx,dt,q,ib,ir,im,iss,gamma)
    elif it ==1:  #Third-order RK
        q = tvdrk3(nx,dx,dt,q,ib,ir,im,iss,gamma)
    else: #4th order RK
        q = crk4(nx,dx,dt,q,ib,ir,im,iss,gamma)
    
    print(k,time,dt)
    fname = 'Data/cell_k='+str(k)+'.npz'
    #writing results
    outresult(fname,nx,dx,time,q)
    if iend == 1:
        break 


outresult('Data/final.npz',nx,dx,time,q)
