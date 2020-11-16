# -*- coding: utf-8 -*-
"""
Learning from past and present (LPP) closure for 1D Shock-tube [Sod's problem]
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
from numpy import linalg as LA
from scipy.linalg import block_diag

from numpy.random import seed
seed(0)

import tensorflow as tf
tf.random.set_seed(1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import load_model

from sklearn.preprocessing import MinMaxScaler
import joblib

import os
import sys
#%% Define Functions

#-----------------------------------------------------------------------------#
# Neural network Routines
#-----------------------------------------------------------------------------#

def create_training_data_lstm(features,labels, m, n, lookback):
    # m : number of snapshots 
    # n: number of states
    ytrain = [labels[i,:] for i in range(lookback,m)]
    ytrain = np.array(ytrain)    
    
    npt = int((n-1)/4)
    xtrain = np.zeros((m-lookback,lookback,7*npt-7))
    for i in range(m-lookback):
        #a = np.copy(features[i,:])
        a = np.copy(np.hstack([features[i,:], \
                       features[i+1,3:npt],\
                       features[i+1,npt+3:2*npt],\
                       features[i+1,2*npt+3:3*npt],
                       features[i+1,-1] ]))
        for j in range(1,lookback):
            tmp = np.copy( np.hstack([features[i+j,:], \
                       features[i+j+1,3:npt],\
                       features[i+j+1,npt+3:2*npt],\
                       features[i+j+1,2*npt+3:3*npt],
                       features[i+j+1,-1] ]) )
            a = np.vstack((a,tmp))
        xtrain[i,:,:] = a
    return xtrain , ytrain


#-----------------------------------------------------------------------------!
#compute rhs for numerical solutions
#-----------------------------------------------------------------------------!
 

## Initial condition [problem definition]
## Initial condition [problem definition]
def init(nx,dx):
    
    lx = dx*nx
    x = np.linspace(-3*dx,lx+3*dx,nx+7)
    q = np.zeros([nx+7,3]) #3 ghost points in each direction
        
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
        if x[i] <= x0:
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
def euler1st(nx,dx,dt,q,ib,gamma):
    s = rhs(nx,dx,q,ib,gamma)
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

    s1 = rhs(nx,dx,q,ib,gamma)
    u[3:nx+4,:] = q[3:nx+4,:] + 0.5*dt*s1
    
    s2 = rhs(nx,dx,u,ib,gamma)
    u[3:nx+4,:] = q[3:nx+4,:] + 0.5*dt*s2

    s3 = rhs(nx,dx,u,ib,gamma)
    u[3:nx+4,:] = q[3:nx+4,:] + dt*s3

    s4 = rhs(nx,dx,u,ib,gamma)
    q[3:nx+4,:] = q[3:nx+4,:] + a*dt*(s1+2*s2+2*s3+s4)

    return q

########
## TVD Runge Kutta 3rd order
def tvdrk3L(nx,dx,dt,q,ib,gamma,nxb):
    
    a = 1.0/3.0
    b = 2.0/3.0
    u = np.zeros([nxb-3+7,3])
    u = np.copy(q)#     ql[-3:,:] = q[nxb-2:nxb+1,:]

    #1st step
    s = rhsL(nx,dx,q,ib,gamma,nxb)
    u[3:nxb-3+4,:] = q[3:nxb-3+4,:] + dt*s
    
    #2nd step
    s = rhsL(nx,dx,u,ib,gamma,nxb)
    u[3:nxb-3+4,:] = 0.75*q[3:nxb-3+4,:] + 0.25*u[3:nxb-3+4,:] + 0.25*dt*s

    #3rd step
    s = rhsL(nx,dx,u,ib,gamma,nxb)
    q[3:nxb-3+4,:] = a*q[3:nxb-3+4,:] + b*u[3:nxb-3+4,:] + b*dt*s

    return q
   


## Computing Right Hand Side for left subdomain
def rhsL(nx,dx,q,ib,gamma,nxb):

    # Apply boundary conditions
    if ib ==1: # 1st order non-reflective (transmissive)
        q[2,:]  = q[3,:]
        q[1,:]  = q[3,:]
        q[0,:]  = q[3,:]
    
    elif ib==2: # 2nd order non-reflective (transmissive)
    
        q[2,:]  = 2.0*q[3,:] - q[4,:] 
        q[1,:]  = 2.0*q[2,:] - q[3,:] 
        q[0,:]  = 2.0*q[1,:] - q[2,:] 
    
    else: # 3rd order non-reflective (transmissive)
    
        q[2,:]  = 3.0*q[3,:] - 3.0*q[4,:] + q[5,:] 
        q[1,:]  = 3.0*q[2,:] - 3.0*q[3,:] + q[4,:]
        q[0,:]  = 3.0*q[1,:] - 3.0*q[2,:] + q[3,:] 

    	
    # Reconstruction scheme
    qL,qR = weno5(nxb-3,q)

    fL = flux(nxb-3,qL,gamma)
    fR = flux(nxb-3,qR,gamma)

    # Spectral radius of Jacobian
    rad = np.zeros(nxb-3+5)
    for i in range(2,nxb-3+5):
        l1 = np.abs(q[i,1]/q[i,0])
        l2 = np.abs(q[i,1]/q[i,0] + np.sqrt(gamma*((gamma-1.0)*(q[i,2]-0.5*q[i,1]*q[i,1]/q[i,0]))/q[i,0]))
        l3 = np.abs(q[i,1]/q[i,0] - np.sqrt(gamma*((gamma-1.0)*(q[i,2]-0.5*q[i,1]*q[i,1]/q[i,0]))/q[i,0]))
        rad[i] = np.max([l1,l2,l3])
    
    # Propagation speed
    ps = np.zeros(nxb-3+4)
    for i in range(2,nxb-3+4):
        ps[i] = np.max([rad[i],rad[i+1]])

    # Compute fluxes with Rusanov
    f = np.zeros([nxb-3+4,3])
    for i in range(2,nxb-3+4):
    	f[i,:] = 0.5*((fR[i,:]+fL[i,:]) - ps[i]*(qR[i,:]-qL[i,:]))

    # Compute rhs
    s = np.zeros([nxb-3+4,3])
    for i in range(3,nxb-3+4):
    	s[i,:] = -(f[i,:]-f[i-1,:])/dx
           
    return s[3:nxb-3+4]


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

#export results for the left sub-domain
def outresultL(fname,nxb,dx,time,q):
    
    lx = dx*nxb
    x = np.linspace(-3*dx,lx,nxb+4)

    #computing results at cell centers:
    r = np.zeros(nxb+4)
    u = np.zeros(nxb+4)
    e = np.zeros(nxb+4)
    p = np.zeros(nxb+4)
    h = np.zeros(nxb+4)
    a = np.zeros(nxb+4)
    m = np.zeros(nxb+4)
    for i in range(3,nxb+4):
        r[i]= q[i,0]                        #rho: density 
        u[i]= q[i,1]/q[i,0]                 #u: velocity
        e[i]= q[i,2]/q[i,0]                 #e: internal energy
        p[i]= (gamma-1.0)*(q[i,2]-0.5*q[i,1]*q[i,1]/q[i,0]) #p: pressure
        h[i]= e[i] + p[i]/r[i]              #h: static enthalpy
        a[i]= np.sqrt(gamma*p[i]/r[i])      #a: speed of sound
        m[i]= u[i]/a[i]                     #m: Mach number
    
    
    #writing results at cell centers:
    np.savez(fname, time=time, x=x[3:nxb+4], r=r[3:nxb+4],\
                          u=u[3:nxb+4], p=p[3:nxb+4], e=e[3:nxb+4],\
                          h=h[3:nxb+4], a=a[3:nxb+4], m=m[3:nxb+4])

    return 

## reading data
def inresult(fname):
    
    lx = dx*nx
    #x = np.linspace(-3*dx,lx+3*dx,nx+7)

    #computing results at cell centers:
    r = np.zeros(nx+1)
    u = np.zeros(nx+1)
    e = np.zeros(nx+1)
    p = np.zeros(nx+1)
    h = np.zeros(nx+1)
    a = np.zeros(nx+1)
    m = np.zeros(nx+1)
  
    
    #reading results at cell centers:
    data = np.load(fname)
    time=data['time']
    x = data['x']
    r = data['r']
    u = data['u']
    p = data['p']
    e = data['e']
    h = data['h']
    a = data['a']
    
    m = data['m']
    
    q = np.zeros([nx+1,3])
    q[:,0] = r
    q[:,1] = r*u
    q[:,2] = r*e
    return q


#%% Main program:
    
# Inputs

nx = 400        #nx;
cfl = 0.5   	#cfl;
it = 1      	#it;[0]Euler1st,[1]RK3,[2]RK4
ib = 1      	#ib;[1]1st,[2]2nd,[3]3rd
nt = 90000000   #nt;maximum number of time step

lx = 1
dx = lx/nx
xx = np.linspace(0,lx,nx+1)
nxb = int(nx/2)

training = 'true'
#%%

# create folder
if os.path.isdir("./Data"):
    print('Data folder already exists')
else: 
    print('Creating Data folder')
    os.makedirs("./Data")    
        
# Initial condition
tmax, gamma, q = init(nx,dx)
dtmin = 1e-4
ns = int(tmax/dtmin)

#%% Read data

npt = 3 #number of points in input

npt = npt+2 #(to consider 3 ghost points)
xi = np.zeros((ns+1,4*npt+1))
yi = np.zeros((ns+1,9))

for k in range(ns+1):
    fname = 'Data/cell_k='+str(k)+'.npz'
    q = inresult(fname)
    
    for jj in range(npt):
        xi[k,jj] = q[nxb-jj,0]
        xi[k,npt+jj] = q[nxb-jj,1]
        xi[k,2*npt+jj] = q[nxb-jj,2]
        xi[k,3*npt+jj] = xx[nxb-jj] #x[[nxb,nxb+1,nxb+2]]
        
    xi[k,-1] = k*dtmin

    yi[k,0:3] = q[nxb-2:nxb+1,0]
    yi[k,3:6] = q[nxb-2:nxb+1,1]
    yi[k,6:9] = q[nxb-2:nxb+1,2]


features = xi 
labels = yi

lookback = 1

#%
if training == 'true': 
    xtrain, ytrain = create_training_data_lstm(features, labels,  features.shape[0], \
                                          features.shape[1], lookback)
    
    # Scaling data
    m,n = ytrain.shape # m is number of training samples, n is number of output features 
    scalerOut = MinMaxScaler(feature_range=(-1,1))
    scalerOut = scalerOut.fit(ytrain)
    ytrain = scalerOut.transform(ytrain)
    
    for k in range(lookback):
        if k == 0:
            tmp = xtrain[:,k,:]
        else:
            tmp = np.vstack([tmp,xtrain[:,k,:]])
            
    scalerIn = MinMaxScaler(feature_range=(-1,1))
    scalerIn = scalerIn.fit(tmp)
    for i in range(m):
        xtrain[i,:,:] = scalerIn.transform(xtrain[i,:,:])
    
    #%
    # Shuffling data
    perm = np.random.permutation(m)
    xtrain = xtrain[perm,:,:]
    ytrain = ytrain[perm,:]
    

    # create folder
    if os.path.isdir("./LSTM Model"):
        print('LSTM models folder already exists')
    else: 
        print('Creating LSTM models folder')
        os.makedirs("./LSTM Model")
        
        
    # Removing old models
    model_name = 'LSTM Model/LSTM_BC_Closure_LPP'+str(npt)+'.h5'
    if os.path.isfile(model_name):
       os.remove(model_name)
    
    
    # create the LSTM architecture
    model = Sequential()
    #model.add(Dropout(0.2))
    model.add(LSTM(20, input_shape=(lookback, features.shape[1]+3*npt-8), return_sequences=True, activation='tanh'))
    model.add(LSTM(20, input_shape=(lookback, features.shape[1]+3*npt-8), activation='tanh'))
    model.add(Dense(labels.shape[1]))
    
    # compile model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
    
    # run the model
    history = model.fit(xtrain, ytrain, epochs=200, batch_size=64, validation_split=0.20)
    
    # evaluate the model
    scores = model.evaluate(xtrain, ytrain, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    plt.figure()
    epochs = range(1, len(loss) + 1)
    plt.semilogy(epochs, loss, 'b', label='Training loss')
    plt.semilogy(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    filename = 'LSTM Model/loss_LPP.png'
    plt.savefig(filename, dpi = 200)
    plt.show()
    
    
    # Save the model
    model.save(model_name)
    
    # Save the scales
    filename = 'LSTM Model/input_scaler_LPP'+str(npt)+'.save'
    joblib.dump(scalerIn,filename) 
    filename = 'LSTM Model/output_scaler_LPP'+str(npt)+'.save'
    joblib.dump(scalerOut,filename) 


#%% Testing
 
model_name = 'LSTM Model/LSTM_BC_Closure_LPP'+str(npt)+'.h5'
model = load_model(model_name)  

# load scales
filename = 'LSTM Model/input_scaler_LPP'+str(npt)+'.save'
scalerIn = joblib.load(filename)  
filename = 'LSTM Model/output_scaler_LPP'+str(npt)+'.save'
scalerOut = joblib.load(filename) 
    
xTest = xi[:,:]
yTest = yi[:,:]

xtest = np.zeros((1,lookback,7*npt-7))

# Initializing
iend = 0    
time = 0.0
qLSTM = np.zeros([nxb+4,3,ns+1])

for i in range(lookback):
    time = time + dtmin
    temp = np.copy(np.hstack([xTest[i,:], \
                              xTest[i+1,3:npt],\
                              xTest[i+1,npt+3:2*npt],\
                              xTest[i+1,2*npt+3:3*npt],
                              xTest[i+1,-1] ]))
    temp = temp.reshape(1,-1)
    xtest[0,i,:]  = scalerIn.transform(temp) 

    fname = 'Data/cell_k='+str(i)+'.npz'
    qq = inresult(fname)
    qLSTM[3:,:,i] = np.copy(qq[:nxb+1,:])
    
    fname = 'Data/LPPcell_k='+str(i)+'.npz'
    #writing results
    outresultL(fname,nxb,dx,time,qLSTM[:,:,i])
    
#%
# Prediction
for k in range(lookback,ns+1):
    

    dt = dtmin
    #check for final time step
    if (time+dt >=tmax) :
        dt = tmax-time
        iend=1
    
    time = time + dt
    
    # Internal points
    qLSTM[:,:,k] = tvdrk3L(nx,dx,dt,np.copy(qLSTM[:,:,k-1]),ib,gamma,nxb)

    
    #Last points
    for ii in range(lookback):
        
        tmp = np.copy(np.hstack([xTest[k-lookback+ii,:], \
                              xTest[k-lookback+ii+1,3:npt],\
                              xTest[k-lookback+ii+1,npt+3:2*npt],\
                              xTest[k-lookback+ii+1,2*npt+3:3*npt],
                              xTest[k-lookback+ii+1,-1] ]))
        
        tmp[0:npt] = np.copy(qLSTM[nxb+3:nxb+3-npt:-1,0,k-lookback+ii])
        tmp[npt:2*npt] = np.copy(qLSTM[nxb+3:nxb+3-npt:-1,1,k-lookback+ii])
        tmp[2*npt:3*npt] = np.copy(qLSTM[nxb+3:nxb+3-npt:-1,2,k-lookback+ii])
        
        tmp[4*npt+1:5*npt-2] = np.copy(qLSTM[nxb:nxb+3-npt:-1,0,k-lookback+ii+1])
        tmp[5*npt-2:6*npt-5] = np.copy(qLSTM[nxb:nxb+3-npt:-1,1,k-lookback+ii+1])
        tmp[6*npt-5:7*npt-8] = np.copy(qLSTM[nxb:nxb+3-npt:-1,1,k-lookback+ii+1])
    
        tmp = tmp.reshape(1,-1)
        xtest[0,ii,:] = scalerIn.transform(tmp) 

    ytest = model.predict(xtest)
    ytest = scalerOut.inverse_transform(ytest) # rescale 
       
    qLSTM[-3:,0,k] = np.copy(ytest[0,0:3])
    qLSTM[-3:,1,k] = np.copy(ytest[0,3:6])
    qLSTM[-3:,2,k] = np.copy(ytest[0,6:9])

    if np.mod(k,50) == 0:
        print(k,time,dt)
    fname = 'Data/LPPcell_k='+str(k)+'.npz'
    #writing results
    outresultL(fname,nxb,dx,time,qLSTM[:,:,k])
    if iend == 1:
        break 
outresultL('Data/LPPfinal.npz',nxb,dx,time,qLSTM[:,:,k])
    
