# -*- coding: utf-8 -*-
"""
Learning from past (LP) closure for 1D Burgers problem with an initial 
condition of a pulse.
This correpsonds to Example 2 for the following paper:
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

#-----------------------------------------------------------------------------!
#compute rhs for numerical solutions
#  r = -u*u' + nu*u''
#-----------------------------------------------------------------------------!
def rhsR(nx,dx,nu1,nu2,g1,g2,nxb,u):
    r = np.zeros(nx-nxb+1)
    r[1:nx-nxb] = (nu2/(dx*dx))*(u[2:nx-nxb+1] - 2.0*u[1:nx-nxb] + u[0:nx-nxb-1]) \
                - g2*u[1:nx-nxb] \
                -(1.0/3.0)*(u[2:nx-nxb+1]+u[0:nx-nxb-1]+u[1:nx-nxb])\
                *(u[2:nx-nxb+1]-u[0:nx-nxb-1])/(2.0*dx)
    return r

#-----------------------------------------------------------------------------#
# Neural network Routines
#-----------------------------------------------------------------------------#
def create_training_data_lstm(features,labels, m, n, lookback):
    # m : number of snapshots 
    # n: number of states
    ytrain = [labels[i,:] for i in range(lookback,m)]
    ytrain = np.array(ytrain)    
    
    xtrain = np.zeros((m-lookback,lookback,n))
    for i in range(m-lookback):
        a = np.copy(features[i,:])
        for j in range(1,lookback):
            a = np.vstack((a,features[i+j,:]))
        xtrain[i,:,:] = a
    return xtrain , ytrain


#%% Main program:
    
# Inputs
nx =  4*1024  #spatial resolution
lx = 1.0    #spatial domain
dx = lx/nx
x = np.linspace(0, lx, nx+1)

xb = 0.3
nxb= round(nx*xb)

nu1 = 1e-2   #control dissipation
nu2 = 1e-4   #control dissipation
g1 = 0 #friction
g2 = 1 #friction

tm = 1        #maximum time
dt = 2.5e-4   #solver timestep
nt = round(tm/dt)
t = np.linspace(0, tm, nt+1)

ns = 4000 #number of snapshots to save
freq = round(nt/ns)

training = 'true'
#%% Read data
npt = 3 #number of points in input

pw = [0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26]

uFOM = np.zeros((len(pw),ns+1,nx+1))
xi = np.zeros((len(pw),ns+1,1*npt+2))
yi = np.zeros((len(pw),ns+1,1))

for ii in range(len(pw)):

    data = np.load('./Data/uFOM_xb=' + str(xb)+ '_pw=' + str(pw[ii]) + '_.npy')
    uFOM[ii,:,:] = data.T
    for jj in range(npt):
        xi[ii,:,jj] = data[nxb+jj,:].T

    xi[ii,:,-2] = t
    xi[ii,:,-1] = pw[ii]
    yi[ii,:,0] = data[nxb,:]

#%% Divide into training and testing
xTrain = xi[[0,2,4,6],:,:]
yTrain = yi[[0,2,4,6],:,:]

lookback = 3
#%%
if training == 'true': 
     
    for i in range(4):
        features = xTrain[i,:,:] 
        labels = yTrain[i,:,:]
        xt, yt = create_training_data_lstm(features, labels,  features.shape[0], \
                                          features.shape[1], lookback)
        if i == 0:
            xtrain = xt
            ytrain = yt
        else:
            xtrain = np.vstack((xtrain,xt))
            ytrain = np.vstack((ytrain,yt))
            
    #%%          
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
    
    #%%
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
    model_name = 'LSTM Model/LP_Closure_'+str(npt)+'.h5'
    if os.path.isfile(model_name):
       os.remove(model_name)
    
    
    # create the LSTM architecture
    model = Sequential()
    #model.add(LSTM(20, input_shape=(lookback, features.shape[1]), return_sequences=True, activation='tanh'))
    #model.add(LSTM(30, input_shape=(lookback, features.shape[1]), return_sequences=True, activation='tanh'))
    model.add(LSTM(15, input_shape=(lookback, features.shape[1]), activation='tanh'))
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
    filename = 'LSTM Model/LP_loss.png'
    plt.savefig(filename, dpi = 200)
    plt.show()
    
    
    # Save the model
    model.save(model_name)
    
    # Save the scales
    filename = 'LSTM Model/LP_input_scaler_'+str(npt)+'.save'
    joblib.dump(scalerIn,filename) 
    filename = 'LSTM Model/LP_output_scaler_'+str(npt)+'.save'
    joblib.dump(scalerOut,filename) 


#%% Testing
    
model_name = 'LSTM Model/LP_Closure_'+str(npt)+'.h5'
model = load_model(model_name)  

# load scales
filename = 'LSTM Model/LP_input_scaler_'+str(npt)+'.save'
scalerIn = joblib.load(filename)  
filename = 'LSTM Model/LP_output_scaler_'+str(npt)+'.save'
scalerOut = joblib.load(filename) 

uLSTM = np.zeros((len(pw),ns+1,nx+1))

for kk in range(len(pw)):
    
    xTest = xi[kk,:,:]
    yTest = yi[kk,:,:]    
    xtest = np.zeros((1,lookback,1*npt+2))
    
    # Initializing
    uu = np.zeros(nx-nxb+1)
    u1 = np.zeros(nx-nxb+1)
    uu = np.copy(uFOM[kk,0,nxb:])
    u1 = np.copy(uFOM[kk,0,nxb:])    
    
    uLSTM[kk,0,nxb:] = uu
                
    for i in range(lookback):
        temp = xTest[i,:]
        temp = temp.reshape(1,-1)
        xtest[0,i,:]  = scalerIn.transform(temp) 
        
    # Prediction
    for i in range(lookback,ns+1):
        ytest = model.predict(xtest)
        ytest = scalerOut.inverse_transform(ytest) # rescale  
        
        # integrate one time step || RK3 scheme
        # first step        
        rr = rhsR(nx,dx,nu1,nu2,g1,g2,nxb,uu)
        u1[1:nx-nxb] = uu[1:nx-nxb] + dt*rr[1:nx-nxb]
        
        # second step
        rr = rhsR(nx,dx,nu1,nu2,g1,g2,nxb,u1)
        u1[1:nx-nxb] = 0.75*uu[1:nx-nxb] + 0.25*u1[1:nx-nxb] + 0.25*dt*rr[1:nx-nxb]
        	
        # third step
        rr = rhsR(nx,dx,nu1,nu2,g1,g2,nxb,u1)
        uu[1:nx-nxb] = 1.0/3.0*uu[1:nx-nxb] + 2.0/3.0*u1[1:nx-nxb] + 2.0/3.0*dt*rr[1:nx-nxb]
        
        uu[0] = ytest
        u1[0] = ytest
                     
        uLSTM[kk,i,nxb:] = uu
        print([kk,i])
              
        # Update xtest
        for k in range(lookback-1):
            xtest[0,k,:] = xtest[0,k+1,:]
        tmp = np.copy(xTest[i,:])
        tmp[0:npt] = uu[0:npt] 
        tmp = tmp.reshape(1,-1)
        xtest[0,lookback-1,:] = scalerIn.transform(tmp) 

np.save('./Data/uFOM.npy',uFOM)
np.save('./Data/uLP_'+str(npt)+'.npy',uLSTM)


