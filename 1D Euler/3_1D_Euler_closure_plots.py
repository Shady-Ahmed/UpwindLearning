# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 14:36:54 2020
@author: Shady
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
ip = 1          #ip;[1]Sod,[2]Lax,[3]123,[4]HM,[5]M3,[6]2S,[7]L,[8]MC,[9]peak,[10]SO
iss = 5         #iss;[1]CHARM,[5]minmod,[6]MC,[8]ospre,[10]superbee,[15]vanLeer
it = 1      	#it;[0]Euler1st,[1]RK3,[2]RK4
ir = 3      	#ir;[1]MUSCL,[2]WENO3,[3]WENO5
im = 1      	#im;[1]1st,[2]up,[3]Fromm,[4]3rd,[5]cen,[6]KT
ib = 1      	#ib;[1]1st,[2]2nd,[3]3rd
nt = 90000000   #nt;maximum number of time step

lx = 1

dx = lx/nx

xx = np.linspace(0,lx,nx+1)
nxb = int(nx/2)

npt=3

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


ax[0].set_title(r'{\fontsize{22pt}{3em}\bf{velocity}} \bf{(LP)}', fontsize = 20)
ax[1].set_title(r'{\fontsize{22pt}{3em}\bf{density}} \bf{ (LP)}', fontsize = 20)
ax[2].set_title(r'{\fontsize{22pt}{3em}\bf{pressure}} \bf{ (LP)}', fontsize = 20)

ax[3].set_title(r'{\fontsize{22pt}{3em}\bf{velocity}} \bf{ (LPP)}', fontsize = 20)
ax[4].set_title(r'{\fontsize{22pt}{3em}\bf{density}} \bf{ (LPP)}', fontsize = 20)
ax[5].set_title(r'{\fontsize{22pt}{3em}\bf{pressure}} \bf{ (LPP)}', fontsize = 20)


for i in range(6):
    ax[i].set_xlabel(r'$x$', fontsize = 26)
    ax[i].set_xticks([0.0,0.5,1.0])
    ax[i].set_yticks([0.0,0.5,1.0])


ax[0].set_ylabel(r'$u(x,t)$', fontsize = 26)
ax[3].set_ylabel(r'$u(x,t)$', fontsize = 26)

ax[1].set_ylabel(r'$\rho(x,t)$', fontsize = 26)
ax[4].set_ylabel(r'$\rho(x,t)$', fontsize = 26)

ax[2].set_ylabel(r'$p(x,t)$', fontsize = 26)
ax[5].set_ylabel(r'$p(x,t)$', fontsize = 26)

    

ax[4].legend([r'\bf{True}',r'\bf{LSTM BC Closure}'],loc="center", bbox_to_anchor=(0.5,-0.5),ncol =2,fontsize=22)

        
fig.subplots_adjust(bottom=0.15, hspace=0.55, wspace=0.45)

#plt.savefig('fig5.pdf', dpi = 500, bbox_inches = 'tight')
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
        'size'   : 24}

matplotlib.rc('font', **font)
matplotlib.rcParams['mathtext.default'] = 'it'


t= np.linspace(0,0.2,2001)

ii = 200
fig, ax = plt.subplots(nrows=1,ncols=3, figsize=(12,3))
ax = ax.flat        
ax[0].plot(t,u[ii,:],'k-.',label=r'True',linewidth=3)
ax[1].plot(t,r[ii,:],'k-.',label=r'True',linewidth=3)
ax[2].plot(t,p[ii,:],'k-.',label=r'True',linewidth=3)

ax[0].plot(t,u[ii-1,:],'r--',label=r'True',linewidth=2)
ax[1].plot(t,r[ii-1,:],'r--',label=r'True',linewidth=2)
ax[2].plot(t,p[ii-1,:],'r--',label=r'True',linewidth=2)


ax[0].plot(t,u[ii+1,:],'b-',label=r'True',linewidth=2)
ax[1].plot(t,r[ii+1,:],'b-',label=r'True',linewidth=2)
ax[2].plot(t,p[ii+1,:],'b-',label=r'True',linewidth=2)


ax[1].legend([r'$x_0$',r'$x_0-\Delta x$', r'$x_0+\Delta x$'],loc="center",\
             bbox_to_anchor=(0.5,-0.55),ncol =3,fontsize=24)


for i in range(3):
    ax[i].set_xlabel(r'$t$', fontsize = 26)
    
ax[0].set_title(r'\bf{velocity}', fontsize = 26)
ax[1].set_title(r'\bf{density}', fontsize = 26)
ax[2].set_title(r'\bf{pressure}', fontsize = 26)

ax[0].set_ylabel(r'$u(x,t)$', fontsize = 26)
ax[1].set_ylabel(r'$\rho(x,t)$', fontsize = 26)
ax[2].set_ylabel(r'$p(x,t)$', fontsize = 26)
        
fig.subplots_adjust(hspace=0.4, wspace=0.5)

#plt.savefig('interface.pdf', dpi = 500, bbox_inches = 'tight')
plt.show()


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


t= np.linspace(0,0.2,2001)

ii = 200
fig, ax = plt.subplots(nrows=1,ncols=3, figsize=(11,3))
ax = ax.flat        
ax[0].plot(t,u[ii,:],'k',label=r'True',linewidth=3)
ax[1].plot(t,u[ii,:]-a[ii,:],'k',label=r'True',linewidth=3)
ax[2].plot(t,u[ii,:]+a[ii,:],'k',label=r'True',linewidth=3)

ax[0].plot(t,u[ii-1,:],'r--',label=r'True',linewidth=2)
ax[1].plot(t,u[ii-1,:]-a[ii-1,:],'r--',label=r'True',linewidth=2)
ax[2].plot(t,u[ii-1,:]+a[ii-1,:],'r--',label=r'True',linewidth=2)


ax[0].plot(t,u[ii+1,:],'b-.',label=r'True',linewidth=2)
ax[1].plot(t,u[ii+1,:]-a[ii+1,:],'b-.',label=r'True',linewidth=2)
ax[2].plot(t,u[ii+1,:]+a[ii+1,:],'b-.',label=r'True',linewidth=2)

ax[1].legend([r'$x_0$',r'$x_0-\Delta x$', r'$x_0+\Delta x$'],loc="center",\
             bbox_to_anchor=(0.5,-0.35),ncol =3,fontsize=18)



for i in range(3):
    ax[i].set_xlabel(r'$t$', fontsize = 14)
    
# ax[0].set_title(r'$u$', fontsize = 16)
# ax[1].set_title(r'$u-a$', fontsize = 16)
# ax[2].set_title(r'$u+a$', fontsize = 16)

ax[0].set_ylabel(r'$u(x,t)$', fontsize = 14)
ax[1].set_ylabel(r'$u(x,t)-a(x,t)$', fontsize = 14)
ax[2].set_ylabel(r'$u(x,t)+a(x,t)$', fontsize = 14)
        
fig.subplots_adjust(hspace=0.4, wspace=0.4)

#plt.savefig('./euler_interface.png', dpi = 500, bbox_inches = 'tight')
plt.show()




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
#cbar_ax = fig.add_axes([0.125, 0.47, 0.19, 0.035])
#fig.colorbar(axx, cax=cbar_ax, ticks=ctick, orientation='horizontal',aspect=10)
ax[0].plot([0.5,0.5],[0,0.2],'-.',linewidth=3.5,color='tab:green')


lv0 = -1.5
lv1 = 1.5
nlvls = 12
vv = np.linspace(lv0, lv1, nlvls, endpoint=True)
ctick = np.linspace(lv0, lv1, 3, endpoint=True)

axx = ax[1].contour(x,t,(u-a).T,vv,cmap=mapp, linewidths=2.5)
axx.set_clim([lv0, lv1])
#cbar_ax = fig.add_axes([0.415, 0.47, 0.19, 0.035])
#fig.colorbar(axx, cax=cbar_ax, ticks=ctick, orientation='horizontal',aspect=10)
ax[1].plot([0.5,0.5],[0,0.2],'-.',linewidth=3.5,color='tab:green')

lv0 = -1.5
lv1 = 1.5
nlvls = 20
vv = np.linspace(lv0, lv1, nlvls, endpoint=True)
ctick = np.linspace(lv0, lv1, 3, endpoint=True)

axx = ax[2].contour(x,t,(u+a).T,vv,cmap=mapp, linewidths=2.5)
axx.set_clim([lv0, lv1])
#cbar_ax = fig.add_axes([0.706, 0.47, 0.19, 0.035])
#fig.colorbar(axx, cax=cbar_ax, ticks=ctick, orientation='horizontal',aspect=10)
cbar_ax = fig.add_axes([0.31, 0.47, 0.4, 0.035])
fig.colorbar(axx, cax=cbar_ax, ticks=ctick, orientation='horizontal',aspect=10)

ax[2].plot([0.5,0.5],[0,0.2],'-.',linewidth=3.5,color='tab:green')



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

ax[4].legend([r'$x_0$',r'$x_0-\Delta x$', r'$x_0+\Delta x$'],loc="center",\
             bbox_to_anchor=(0.5,-0.50),ncol =3,fontsize=24)


for i in range(3,6):
    ax[i].set_xlabel(r'$t$', fontsize = 26)
    

ax[3].set_ylabel(r'$u$', fontsize = 26)
ax[4].set_ylabel(r'$u-a$', fontsize = 26)
ax[5].set_ylabel(r'$u+a$', fontsize = 26)
        
fig.subplots_adjust(hspace=0.7, wspace=0.5)

#plt.savefig('charact.pdf', dpi = 500, bbox_inches = 'tight')


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


fig, ax = plt.subplots(nrows=3,ncols=2, figsize=(8,10))
ax = ax.flat     
mapp='seismic'


lv0 = -1.2
lv1 = 1.2
nlvls = 25
vv = np.linspace(lv0, lv1, nlvls, endpoint=True)
ctick = np.linspace(lv0, lv1, 3, endpoint=True)
#ctick[1] = 0.6

axx = ax[0].contour(x,t,(u).T,vv,cmap=mapp)
axx.set_clim([lv0, lv1])
#cbar_ax = fig.add_axes([0.125, 0.47, 0.19, 0.035])
fig.colorbar(axx, ax=ax[0], ticks=ctick, orientation='vertical',aspect=10)
ax[0].plot([0.5,0.5],[0,0.2],'-.k',linewidth=4)


#lv0 = -1.2
#lv1 = 0.2
nlvls = 51
vv = np.linspace(lv0, lv1, nlvls, endpoint=True)
ctick = np.linspace(lv0, lv1, 3, endpoint=True)

axx = ax[2].contour(x,t,(u-a).T,vv,cmap=mapp)
axx.set_clim([lv0, lv1])
#cbar_ax = fig.add_axes([0.415, 0.47, 0.19, 0.035])
#fig.colorbar(axx, cax=cbar_ax, ticks=ctick, orientation='horizontal',aspect=10)
fig.colorbar(axx, ax=ax[2], ticks=ctick, orientation='vertical',aspect=10)

ax[2].plot([0.5,0.5],[0,0.2],'-.k',linewidth=4)

#lv0 = 1.1
#lv1 = 2.1
nlvls = 51
vv = np.linspace(lv0, lv1, nlvls, endpoint=True)
ctick = np.linspace(lv0, lv1, 3, endpoint=True)

axx = ax[4].contour(x,t,(u+a).T,vv,cmap=mapp)
axx.set_clim([lv0, lv1])
#cbar_ax = fig.add_axes([0.706, 0.47, 0.19, 0.035])
#fig.colorbar(axx, cax=cbar_ax, ticks=ctick, orientation='horizontal',aspect=10)
fig.colorbar(axx, ax=ax[4], ticks=ctick, orientation='vertical',aspect=10)
ax[4].plot([0.5,0.5],[0,0.2],'-.k',linewidth=4)



ax[0].set_title(r'$u$', fontsize = 26)
ax[2].set_title(r'$u-a$', fontsize = 26)
ax[4].set_title(r'$u+a$', fontsize = 26)


for i in [0,2,4]:
    ax[i].set_xlabel(r'$x$', fontsize = 26)
    ax[i].set_ylabel(r'$t$', fontsize = 26)




ii = 200

ax[1].plot(t,u[ii,:],'-.k',label=r'True',linewidth=3)
ax[3].plot(t,u[ii,:]-a[ii,:],'-.k',label=r'True',linewidth=3)
ax[5].plot(t,u[ii,:]+a[ii,:],'-.k',label=r'True',linewidth=3)

ax[1].plot(t,u[ii-1,:],'r--',label=r'True',linewidth=2)
ax[3].plot(t,u[ii-1,:]-a[ii-1,:],'r--',label=r'True',linewidth=2)
ax[5].plot(t,u[ii-1,:]+a[ii-1,:],'r--',label=r'True',linewidth=2)


ax[1].plot(t,u[ii+1,:],'b-',label=r'True',linewidth=2)
ax[3].plot(t,u[ii+1,:]-a[ii+1,:],'b-',label=r'True',linewidth=2)
ax[5].plot(t,u[ii+1,:]+a[ii+1,:],'b-',label=r'True',linewidth=2)

ax[5].legend([r'$x_0$',r'$x_0-\Delta x$', r'$x_0+\Delta x$'],loc="center",\
             bbox_to_anchor=(-0.5,-0.75),ncol =3,fontsize=24)

#ax[1].legend([r'$x_0$',r'$x_0-\Delta x$', r'$x_0+\Delta x$'],ncol =1,fontsize=20)

for i in [1,3,5]:
    ax[i].set_xlabel(r'$t$', fontsize = 26)
    
ax[1].set_ylabel(r'$u$', fontsize = 26)
ax[3].set_ylabel(r'$u-a$', fontsize = 26)
ax[5].set_ylabel(r'$u+a$', fontsize = 26)
        
fig.subplots_adjust(hspace=0.7, wspace=0.8)

plt.savefig('charact2.pdf', dpi = 500, bbox_inches = 'tight')
#%%
# gamma = 1.4
# c1 = p/r**gamma
# c2 = u-(2*a)/(gamma-1) 
# c3 = u+(2*a)/(gamma-1) 

# #mask = np.logical_and((c1>1.836), (c1<1.838)) 
# mask = c2 == c2[200,0]
# for i in range(400):
#     for j in range(1,1000):
#         if mask[i,j]:
#             plt.plot(x[i],t[j],'*')
# #%%

# c1 = u*t + u[:,0].reshape([-1,1])
# c2 = (u-a)*t + (u[:,0] - a[:,0]).reshape([-1,1])
# c3 = (u+a)*t + (u[:,0] + a[:,0]).reshape([-1,1])


# #%%
# fig, ax = plt.subplots(nrows=1,ncols=3, figsize=(13,5))
# ax = ax.flat     
# mapp='seismic'

# axx = ax[0].contour(x,t,(c1).T,13,cmap=mapp)
# cbar = fig.colorbar(axx, ax=ax[0], orientation='horizontal',aspect=10)
# cbar.ax.tick_params(labelsize=10)

# axx = ax[1].contour(x,t,(c2).T,15,cmap=mapp)
# cbar = fig.colorbar(axx, ax=ax[1], orientation='horizontal',aspect=10)
# cbar.ax.tick_params(labelsize=10)

# axx = ax[2].contour(x,t,(c3).T,15,cmap=mapp)
# cbar = fig.colorbar(axx, ax=ax[2], orientation='horizontal',aspect=10)
# cbar.ax.tick_params(labelsize=10)


# ax[0].set_title(r'$u$', fontsize = 16)
# ax[1].set_title(r'$u-a$', fontsize = 16)
# ax[2].set_title(r'$u+a$', fontsize = 16)


# for i in range(3):
#     ax[i].set_xlabel(r'$x$', fontsize = 16)
#     ax[i].set_ylabel(r'$t$', fontsize = 16)

# fig.subplots_adjust(hspace=0.4, wspace=0.35)

# #plt.savefig('./charac.png', dpi = 500, bbox_inches = 'tight')


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
#cbar_ax = fig.add_axes([0.125, 0.47, 0.19, 0.035])
#fig.colorbar(axx, cax=cbar_ax, ticks=ctick, orientation='horizontal',aspect=10)
ax[0].plot([0.5,0.5],[0,0.2],'-.',linewidth=3.5,color='k')


lv0 = -1.5
lv1 = 1.5
nlvls = 12
vv = np.linspace(lv0, lv1, nlvls, endpoint=True)
ctick = np.linspace(lv0, lv1, 3, endpoint=True)

axx = ax[1].contour(x,t,(u-a).T,vv,cmap=mapp, linewidths=2.5)
axx.set_clim([lv0, lv1])
#cbar_ax = fig.add_axes([0.415, 0.47, 0.19, 0.035])
#fig.colorbar(axx, cax=cbar_ax, ticks=ctick, orientation='horizontal',aspect=10)
ax[1].plot([0.5,0.5],[0,0.2],'-.',linewidth=3.5,color='k')

lv0 = -1.5
lv1 = 1.5
nlvls = 20
vv = np.linspace(lv0, lv1, nlvls, endpoint=True)
ctick = np.linspace(lv0, lv1, 3, endpoint=True)

axx = ax[2].contour(x,t,(u+a).T,vv,cmap=mapp, linewidths=2.5)
axx.set_clim([lv0, lv1])
#cbar_ax = fig.add_axes([0.706, 0.47, 0.19, 0.035])
#fig.colorbar(axx, cax=cbar_ax, ticks=ctick, orientation='horizontal',aspect=10)
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

plt.savefig('charact.pdf', dpi = 500, bbox_inches = 'tight')
plt.savefig('fig6.pdf', dpi = 500, bbox_inches = 'tight')


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


t= np.linspace(0,0.2,2001)

ii = 200
fig, ax = plt.subplots(nrows=1,ncols=3, figsize=(12,3))
ax = ax.flat        
ax[0].plot(t,u[ii,:],'k-.',label=r'True',linewidth=3)
ax[1].plot(t,r[ii,:],'k-.',label=r'True',linewidth=3)
ax[2].plot(t,p[ii,:],'k-.',label=r'True',linewidth=3)

ax[0].plot(t,u[ii-1,:],'r--',label=r'True',linewidth=2)
ax[1].plot(t,r[ii-1,:],'r--',label=r'True',linewidth=2)
ax[2].plot(t,p[ii-1,:],'r--',label=r'True',linewidth=2)


ax[0].plot(t,u[ii+1,:],'b-',label=r'True',linewidth=2)
ax[1].plot(t,r[ii+1,:],'b-',label=r'True',linewidth=2)
ax[2].plot(t,p[ii+1,:],'b-',label=r'True',linewidth=2)


ax[1].legend([r'$x_b$',r'$x_b-\Delta x$', r'$x_b+\Delta x$'],loc="center",\
             bbox_to_anchor=(0.5,-0.55),ncol =3,fontsize=24)


for i in range(3):
    ax[i].set_xlabel(r'$t$', fontsize = 26)
    ax[i].set_xticks([0,0.1,0.2])

    
#ax[0].set_title(r'\bf{velocity}', fontsize = 26)
#ax[1].set_title(r'\bf{density}', fontsize = 26)
#ax[2].set_title(r'\bf{pressure}', fontsize = 26)

ax[0].set_ylabel(r'$u(x,t)$', fontsize = 26)
ax[1].set_ylabel(r'$\rho(x,t)$', fontsize = 26)
ax[2].set_ylabel(r'$p(x,t)$', fontsize = 26)
        
fig.subplots_adjust(hspace=0.4, wspace=0.5)

plt.savefig('interface.pdf', dpi = 500, bbox_inches = 'tight')
plt.show()

