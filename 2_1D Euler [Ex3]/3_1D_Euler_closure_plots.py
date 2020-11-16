# -*- coding: utf-8 -*-
"""
Plotting scripts for 1D Shock-tube [Sod's problem]
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

from numpy.random import seed
import os
import sys


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

npt=3

t= np.linspace(0,0.2,2001)


# create folder
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


fig, ax = plt.subplots(nrows=2,ncols=3, figsize=(12,8))
ax = ax.flat

i = 2000
fname = 'Data/cell_k='+str(i)+'.npz'
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
    
ax[0].plot(x,u,'b',label=r'True',linewidth=3)
ax[1].plot(x,r,'b',label=r'True',linewidth=3)
ax[2].plot(x,p,'b',label=r'True',linewidth=3)


####LP
fname = 'Data/lPcell_k='+str(i)+'.npz'
data = np.load(fname)
time=data['time']
x = data['x']
r = data['r']
u = data['u']
p = data['p']
e = data['e']
h = data['h']
a = data['a']
    
ax[0].plot(x,u,'--r',label=r'LSTM BC Closure',linewidth=3)
ax[1].plot(x,r,'--r',label=r'LSTM BC Closure',linewidth=3)
ax[2].plot(x,p,'--r',label=r'LSTM BC Closure',linewidth=3)


#### LPP
fname = 'Data/cell_k='+str(i)+'.npz'
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
    
ax[3].plot(x,u,'b',label=r'True',linewidth=3)
ax[4].plot(x,r,'b',label=r'True',linewidth=3)
ax[5].plot(x,p,'b',label=r'True',linewidth=3)


fname = 'Data/lPPcell_k='+str(i)+'.npz'
data = np.load(fname)
time=data['time']
x = data['x']
r = data['r']
u = data['u']
p = data['p']
e = data['e']
h = data['h']
a = data['a']
    
ax[3].plot(x,u,'--r',label=r'LSTM BC Closure',linewidth=3)
ax[4].plot(x,r,'--r',label=r'LSTM BC Closure',linewidth=3)
ax[5].plot(x,p,'--r',label=r'LSTM BC Closure',linewidth=3)


ax[0].set_title(r'{\fontsize{22pt}{3em}$u$} \bf{(LP)}', fontsize = 20)
ax[1].set_title(r'{\fontsize{22pt}{3em}$\rho$} \bf{ (LP)}', fontsize = 20)
ax[2].set_title(r'{\fontsize{22pt}{3em}$p$} \bf{ (LP)}', fontsize = 20)

ax[3].set_title(r'{\fontsize{22pt}{3em}$u$} \bf{(LPP)}', fontsize = 20)
ax[4].set_title(r'{\fontsize{22pt}{3em}$\rho$} \bf{(LPP)}', fontsize = 20)
ax[5].set_title(r'{\fontsize{22pt}{3em}$p$} \bf{(LPP)}', fontsize = 20)


for i in range(6):
    ax[i].set_xlabel(r'$x$', fontsize = 26)
    ax[i].set_xticks([0.0,0.5,1.0])
    ax[i].set_yticks([0.0,0.5,1.0])


ax[0].set_ylabel(r'$u(x,0.2)$', fontsize = 26)
ax[3].set_ylabel(r'$u(x,0.2)$', fontsize = 26)

ax[1].set_ylabel(r'$\rho(x,0.2)$', fontsize = 26)
ax[4].set_ylabel(r'$\rho(x,0.2)$', fontsize = 26)

ax[2].set_ylabel(r'$p(x,0.2)$', fontsize = 26)
ax[5].set_ylabel(r'$p(x,0.2)$', fontsize = 26)

    
ax[4].legend([r'\bf{True}',r'\bf{LSTM BC Closure}'],loc="center", bbox_to_anchor=(0.5,-0.5),ncol =2,fontsize=22)
        
fig.subplots_adjust(bottom=0.15, hspace=0.55, wspace=0.45)

plt.savefig('fig5.pdf', dpi = 500, bbox_inches = 'tight')
plt.show()


#%%
u = np.zeros([nx+1,2001])
a = np.zeros([nx+1,2001])
p = np.zeros([nx+1,2001])
r = np.zeros([nx+1,2001])

for i in range(2001):
     
    fname = 'Data/cell_k='+str(i)+'.npz'

    #reading results at cell centers:
    data = np.load(fname)
    time=data['time']
    
    u[:,i] = data['u']   
    a[:,i] = data['a']
    p[:,i] = data['p']
    r[:,i] = data['r']

x = data['x']   

#%%
import matplotlib

matplotlib.rc('text', usetex=True)

matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']

font = {'family' : 'sans-serif',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)
matplotlib.rcParams['mathtext.default'] = 'it'


fig, ax = plt.subplots(nrows=2,ncols=3, figsize=(12,8.5))
ax = ax.flat     
mapp='jet'
mapp='seismic'


lv0 = -1.5
lv1 = 1.5
nlvls = 12
vv = np.linspace(lv0, lv1, nlvls, endpoint=True)
ctick = np.linspace(lv0, lv1, 3, endpoint=True)
ctick[1] = 0.0

axx = ax[0].contour(x,t,(u).T,vv,cmap=mapp, linewidths=2.5)
axx.set_clim([lv0, lv1])
ax[0].plot([0.5,0.5],[0,0.2],'-.',linewidth=3.5,color='k')


lv0 = -1.5
lv1 = 1.5
nlvls = 12
vv = np.linspace(lv0, lv1, nlvls, endpoint=True)
ctick = np.linspace(lv0, lv1, 3, endpoint=True)

axx = ax[1].contour(x,t,(u-a).T,vv,cmap=mapp, linewidths=2.5)
axx.set_clim([lv0, lv1])
ax[1].plot([0.5,0.5],[0,0.2],'-.',linewidth=3.5,color='k')

lv0 = -1.5
lv1 = 1.5
nlvls = 20
vv = np.linspace(lv0, lv1, nlvls, endpoint=True)
ctick = np.linspace(lv0, lv1, 3, endpoint=True)

axx = ax[2].contour(x,t,(u+a).T,vv,cmap=mapp, linewidths=2.5)
axx.set_clim([lv0, lv1])
cbar_ax = fig.add_axes([0.31, 0.47, 0.4, 0.035])
fig.colorbar(axx, cax=cbar_ax, ticks=ctick, orientation='horizontal',aspect=10)

ax[2].plot([0.5,0.5],[0,0.2],'-.',linewidth=3.5,color='k')


ax[0].set_title(r'$u$', fontsize = 26)
ax[1].set_title(r'$u-a$', fontsize = 26)
ax[2].set_title(r'$u+a$', fontsize = 26)


for i in range(3):
    ax[i].set_xlabel(r'$x$', fontsize = 26)
    ax[i].set_ylabel(r'$t$', fontsize = 26)


ii = 200

ax[3].plot(t,u[ii,:],'-.k',label=r'True',linewidth=3)
ax[4].plot(t,u[ii,:]-a[ii,:],'-.k',label=r'True',linewidth=3)
ax[5].plot(t,u[ii,:]+a[ii,:],'-.k',label=r'True',linewidth=3)

ax[3].plot(t,u[ii-1,:],'r--',label=r'True',linewidth=2)
ax[4].plot(t,u[ii-1,:]-a[ii-1,:],'r--',label=r'True',linewidth=2)
ax[5].plot(t,u[ii-1,:]+a[ii-1,:],'r--',label=r'True',linewidth=2)


ax[3].plot(t,u[ii+1,:],'b-',label=r'True',linewidth=2)
ax[4].plot(t,u[ii+1,:]-a[ii+1,:],'b-',label=r'True',linewidth=2)
ax[5].plot(t,u[ii+1,:]+a[ii+1,:],'b-',label=r'True',linewidth=2)

ax[4].legend([r'$x_b$',r'$x_b-\Delta x$', r'$x_b+\Delta x$'],loc="center",\
             bbox_to_anchor=(0.5,-0.50),ncol =3,fontsize=24)


for i in range(3,6):
    ax[i].set_xlabel(r'$t$', fontsize = 26)
    ax[i].set_xticks([0,0.1,0.2])


ax[3].set_ylabel(r'$u$', fontsize = 26)
ax[4].set_ylabel(r'$u-a$', fontsize = 26)
ax[5].set_ylabel(r'$u+a$', fontsize = 26)
        
fig.subplots_adjust(hspace=0.7, wspace=0.5)

plt.savefig('fig6.pdf', dpi = 500, bbox_inches = 'tight')


