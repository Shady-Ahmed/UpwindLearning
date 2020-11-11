# -*- coding: utf-8 -*-
"""
Plotting script for 1D Burgers problem with an initial condition of pulse
This correpsonds to Example [Figure 4] for the following paper:
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
from numpy import linalg as LA
from scipy.linalg import block_diag

from numpy.random import seed
seed(0)

import os
import sys


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

im = 1 #numerical scheme [0 for compact, 1 for central, and 2 for upwind]


#%% Read data
npt = 3
uFOM = np.load('.Data/uFOM.npy')


uLSTM1= np.load('.Data/uLP_'+str(npt)+'.npy')
uLSTM3= np.load('./Data/uLPP_'+str(npt)+'.npy')


#%% Plotting

# create plot folder
if os.path.isdir("./Plots"):
    print('Plots folder already exists')
else: 
    print('Creating Plots folder')
    os.makedirs("./Plots")
    

#%%

import matplotlib

matplotlib.rc('text', usetex=True)

matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']

font = {'family' : 'sans-serif',
        'weight' : 'bold',
        'size'   : 24}

matplotlib.rc('font', **font)
matplotlib.rcParams['mathtext.default'] = 'it'


ii = ns

fig, ax = plt.subplots(nrows=2,ncols=3, figsize=(12,8))
ax = ax.flat
    

kk = 1
for ii in [0,1000,2000,3000,4000]:

    for i in [0,3]:
        ax[i].plot(x[:],uFOM[kk,ii,:],'b',label=r'True',linewidth=3)
    ax[0].plot(x[nxb:],uLP[kk,ii,nxb:],'--r',label=r'LSTM BC Closure',linewidth=3)   
    ax[3].plot(x[nxb:],uLPP[kk,ii,nxb:],'--r',label=r'LSTM BC Closure',linewidth=3)

    for i in [0,1,2,3,4,5]:   
        ax[i].arrow(0.4, 0.95, 0.24, -0.4, head_width=0.05, head_length=0.07, fc='k', ec='k')
        ax[i].annotate(r"{time}", fontsize=20, xy=(0, 0), xytext=(0.55, 0.75))#, arrowprops=dict(arrowstyle="->"))

kk = 3
for ii in [0,1000,2000,3000,4000]:
    for i in [1,4]:
        ax[i].plot(x[:],uFOM[kk,ii,:],'b',label=r'True',linewidth=3)
    ax[1].plot(x[nxb:],uLP[kk,ii,nxb:],'--r',label=r'LSTM BC Closure',linewidth=3)
    ax[4].plot(x[nxb:],uLPP[kk,ii,nxb:],'--r',label=r'LSTM BC Closure',linewidth=3)


kk = 5
for ii in [0,1000,2000,3000,4000]:
    for i in [2,5]:
        ax[i].plot(x[:],uFOM[kk,ii,:],'b',label=r'True',linewidth=3)
    ax[2].plot(x[nxb:],uLP[kk,ii,nxb:],'--r',label=r'LSTM BC Closure',linewidth=3)
    ax[5].plot(x[nxb:],uLPP[kk,ii,nxb:],'--r',label=r'LSTM BC Closure',linewidth=3)



for i in range(6):
    ax[i].set_xlabel(r'$x$', fontsize=26)
    ax[i].set_xticks([0.0,0.5,1.0])
    ax[i].set_ylabel(r'$u(x,t)$', fontsize=26)
    ax[i].set_yticks([0.0,0.5,1.0])
        
        
fig.subplots_adjust(bottom=0.15, hspace=0.55, wspace=0.45)

ax[0].set_title(r'{\fontsize{26pt}{3em}$w_p$} {\fontsize{22pt}{3em}$ =0.21$} \bf{(LP)}',fontsize=20)
ax[1].set_title(r'{\fontsize{26pt}{3em}$w_p$} {\fontsize{22pt}{3em}$ =0.23$} \bf{(LP)}',fontsize=20)
ax[2].set_title(r'{\fontsize{26pt}{3em}$w_p$} {\fontsize{22pt}{3em}$ =0.25$} \bf{(LP)}',fontsize=20)

ax[3].set_title(r'{\fontsize{26pt}{3em}$w_p$} {\fontsize{22pt}{3em}$ =0.21$}  \bf{(LPP)}',fontsize=20)
ax[4].set_title(r'{\fontsize{26pt}{3em}$w_p$} {\fontsize{22pt}{3em}$ =0.23$}  \bf{(LPP)}',fontsize=20)
ax[5].set_title(r'{\fontsize{26pt}{3em}$w_p$} {\fontsize{22pt}{3em}$ =0.25$} \bf{(LPP)}',fontsize=20)

  
ax[4].legend([r'\bf{True}',r'\bf{LSTM BC Closure}'],loc="center", bbox_to_anchor=(0.5,-0.5),ncol =2,fontsize=22)


plt.savefig('fig4.pdf', dpi = 500, bbox_inches = 'tight')
plt.show()
    
