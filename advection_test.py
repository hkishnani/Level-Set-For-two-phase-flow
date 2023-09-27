"""
Created on Tue Apr 18 13:55:45 2023

@author: himanshu
"""
import numpy as np
from math import pi
from post_process import VISUALIZE_VOL
from tqdm import trange
import LS_advect as LSA
from LS_funcs import H
from plotly.graph_objects import Figure

# %% Setting up uniform mesh
n_pts = 120
x_1 = 0
x_2 = 1
y_1 = 0
y_2 = 1
r = 0.15
h = 0.5
k = 0.75
dx = (x_2 - x_1) / n_pts
dy = dx
dt = 0.1 * dx

T = 6.0 + dt

X_c = np.arange(x_1+dx/2, x_2 + dx/2, dx)

Y_c = np.arange(y_1+dy/2, y_2 + dy/2, dy)

X_grid, Y_grid = np.meshgrid(X_c, Y_c)

nrows = Y_c.size
mcols = X_c.size
# %% The implicit surface
# Defining the implicit surface initialization
phi = lambda x, y, h, k, a: np.sqrt((x - h) ** 2 + (y - k) ** 2) - a

Z = phi(X_grid, Y_grid, h, k, r)

# Generating velocity fields


# %% Initialization details
MAX_T_STEPS = int(T / dt) + 1
phi_0 = Z
bound = dx

eps = 1.5*dx

V0 = (1 - H(phi_0, eps = eps)).sum() * dx**2

# %% Generating Velocities for Vortex in Box
u_VIB = lambda x, y: -np.sin(pi * x) ** 2 * np.sin(2 * pi * y)
v_VIB = lambda x, y: np.sin(pi * y) ** 2 * np.sin(2 * pi * x)

i = 0
n_pics = 2
freq_plot = int(MAX_T_STEPS / n_pics)

fig = Figure()
fig.update_coloraxes(showscale=False)
# %% Looping details
for t_steps in trange(MAX_T_STEPS):
    
    # ---------------PLOTTING-------------------------------------------------
    if (t_steps % freq_plot == 0):
        Vol = (1 - H(phi_0, eps = eps)).sum() * dx**2

        title = f"\n VOL = {Vol} \n % Δ Vol = { ( (Vol - V0) / V0 )*100 } \n Time = {t_steps*dt} sec \n"
        print(title)
        
        fig = VISUALIZE_VOL(phi=phi_0, X = X_c, Y = Y_c,
                      dx = dx, 
                      title = f"Time = {t_steps*dt} sec \n",
                      n_pic=i, fig=fig)
        
        
        with open(f'CIS_mesh{n_pts}_t{int(t_steps*dt)}.npy', 'wb') as f:
            np.save(f, phi_0)
        
        i+=1
    # ---------------PLOTTING-------------------------------------------------
    

    V = [u_VIB(X_grid, Y_grid) * np.cos(pi * (t_steps / T) * dt),
         v_VIB(X_grid, Y_grid) * np.cos(pi * (t_steps / T) * dt)]

    phi_0 = LSA.LS_RK4_WENO5(phi_0=phi_0, u=V[0], v=V[1],
                             dt=dt, dx=dx, dy=dy)

    #--------------------- Start Reinitialization-----------------------------
    reinit_iter = 0
    while True:
        reinit_iter += 1
        phi_0_old = phi_0
        phi_0 = LSA.TVD_RK3_WENO5(phi0=phi_0_old,
                                   dt=dt, dx=dx, dy=dy)

        if np.linalg.norm(phi_0 - phi_0_old) < 1e-5 or reinit_iter > 20:
            break
    #---------------------- End Reinitialization------------------------------

fig = VISUALIZE_VOL(phi=phi_0, X = X_c, Y = Y_c,
              dx = dx, 
              title = f"Time = {t_steps*dt} sec \n",
              n_pic=i, fig=fig)

with open(f'CIS_mesh{n_pts}_t{int(t_steps*dt)}.npy', 'wb') as f:
    np.save(f, phi_0)


Vol = (1 - H(phi_0, eps = eps)).sum() * dx**2
title = f"\n VOL = {Vol} \n % Δ Vol = { ( (Vol - V0) / V0 )*100 } \n Time = {t_steps*dt} sec \n"
print(title)
np.linalg.norm(H(Z, eps) - H(phi_0, eps))