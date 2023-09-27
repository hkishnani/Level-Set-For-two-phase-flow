# SIMPLE Algorithm -> main file
# Air bubble in Water problem

# Predefined libraries -----------------------------
import numpy as np
import scipy as sp
import tqdm
import math
# Predefined libraries -----------------------------

# Custom libraries
import MESH
from momentum import MOM_LINK_ASSEMBLER_FOU, MOMENTUM_INTERPOLATION
from pressure_solver import PPE_SOLVER
from corrections import FACE_VEL_CORRECTION, CC_VEL_CORRECTION
# momentum link assembler with First order upwind convective scheme

import LS_funcs as LSF
import LS_advect as LSA
# Custom libraries

# Post-Processing libraries
import post_process as post

# Post-Processing libraries

#%% Start simulation
# Physical parameters ------------------------------
L_x= 0.01
mcols = 200
L_y = 0.04
nrows = 800
# Physical domain dimensions

h, k  = L_x/2, L_y/8
r = 0.4*L_x
# Location of Circle

u_avg = 0
# External field velocity

T_max = 0.0
# Simulation Time
dt = 0.1*(L_x/mcols)
# time step size

rho_1 = 1      # in Kg/m3
rho_2 = 2      # in Kg/m3 --> Surrounding fluid
mu_1  = 1e-1   # in Pa.s --> bubble fluid
mu_2  = 1e-5   # in Pa.s --> Surrounding fluid

sigma = 0.0  # in N/m
# 1 -> Air,  2 -> Water

g = np.array([0.0, -9.81])  # in m/sec
# Physical parameters ------------------------------

# Meshing Details ----------------------------------
dx, del_x, X_c = MESH.CELL_CENT_UNIFORM(start=0.0, end=L_x, n_pts=mcols)
dy, del_y, Y_c = MESH.CELL_CENT_UNIFORM(start=0.0, end=L_y, n_pts=nrows)
# Take advantage of structured grid

# Location of Finite Difference grid on Vertex centered mesh
X_v = MESH.VERTEX_CENT_UNIFORM(start=0.0, end=L_x, n_pts=mcols+1)
Y_v = MESH.VERTEX_CENT_UNIFORM(start=0.0, end=L_y, n_pts=nrows+1)

# Location of Collocated cells on Cell centered mesh
dx = dx.reshape(1, -1)
del_x = del_x.reshape(1, -1)
X_c = X_c.reshape(1, -1)

dy = dy.reshape(-1, 1)
del_y = del_y.reshape(-1, 1)
Y_c = Y_c.reshape(-1, 1)
# Location of Collocated cells on Cell centered mesh

# Location of vertex centers
X_v = X_v.reshape(1, -1)
Y_v = Y_v.reshape(-1, 1)
# Location of vertex centers

# Meshing Details ----------------------------------

# Boundary Conditions-------------------------------
BC = {'left': 'No_slip_wall',
      'right': 'No_slip_wall',
      'top': 'No_slip_wall',
      'bottom': 'No_slip_wall'}

u_BC = {'left': np.zeros(shape=(nrows, 1)),
        'right': np.zeros(shape=(nrows, 1)),
        'top': u_avg * np.ones(shape=(1, mcols)),
        'bottom': np.zeros(shape=(1, mcols))
        }

v_BC = {'left': np.zeros(shape=(nrows, 1)),
        'right': np.zeros(shape=(nrows, 1)),
        'top': np.zeros(shape=(1, mcols)),
        'bottom': np.zeros(shape=(1, mcols))
        }
# Boundary Conditions-------------------------------

# Derived parameters -------------------------------

# Value of Level set for circular inital patch
# phi_CC = np.ones(shape=(nrows, mcols))
# phi_VC = np.ones(shape=(nrows+1, mcols+1))
phi_CC, phi_VC = LSF.PATCH_CIRCLE(h = h, k = k, r = r,
                                  Xc = X_c, Yc = Y_c,
                                  Xv = X_v, Yv = Y_v)

# Initializing properties using Level Set function
mu  =  mu_1 + ( mu_2 -  mu_1) * LSF.H(phi_CC, eps=1.5 * dx.max())
rho = rho_1 + (rho_2 - rho_1) * LSF.H(phi_CC, eps=1.5 * dx.max())
post.VISUALIZE_FIELD(rho, X_c, Y_c, title=f'Density initial')
# Initializing properties using Level Set function

Re = math.sqrt(8 * r**3 * np.linalg.norm(g))*(rho_2/mu_2)
Bo = 4 * rho_2 * r**2 * np.linalg.norm(g) / sigma
print("\n Re = ", Re, "\n", "\n Bo = ", Bo, )
# Limiting Courant number

# Derived parameters -------------------------------

# Collocated Variable arrangement-------------------
u, u_star, uP_hat = np.zeros(shape=(3, nrows, mcols), dtype=float)
v, v_star, vP_hat = np.zeros(shape=(3, nrows, mcols), dtype=float)
u_prime, v_prime = np.zeros(shape=(2, nrows, mcols), dtype=float)
p, p_star, p_prime = np.zeros(shape=(3, nrows, mcols), dtype=float)
# Collocated Variable arrangement-------------------

# Face velocity arrangement-------------------------
u_e, de_avg = np.zeros(shape=(2, nrows, mcols - 1))
v_n, dn_avg = np.zeros(shape=(2, nrows - 1, mcols))
# Face velocity arrangement-------------------------

# Visualize some things-----------------------------
# post.VISUALIZE_MESH(dx, dy)
# Visualize some things-----------------------------

# Simulation detaiLs--------------------------------
max_mom_iter = 7
max_p_iter = 15
max_m_iter = 10   # Outer loop iteration
PPE_rel_f = 0.2   # PPE norm of next iteration

al_mom = 7e-1
al_p = 1e-5

eps_mass_tol = 1e-9
# Simulation details--------------------------------

t_step = 0
t_0 = 0.0

V0 = (1-LSF.H(phi_CC, eps = 1.5*dx.max())).sum() * dx.max()**2
# Orignal volume

post.INITIALIZE_DOMAIN(phi_CC, X_c, Y_c, 
                           u, v, L_x, L_y, p,
                           "CHANGE IN VOLUME + TIME STAMP", V0)
#%% Chunky slice
T_max += 0.1
t_0 = t_step*dt
for t in tqdm.tqdm(np.mgrid[t_0 : T_max+dt : dt]):

    # Start Solve for velocities using SIMPLE
    for iter_s in range(max_m_iter):

        # Picard's idea for Momentum solving
        d_u, uP_hat, u_star, d_v, vP_hat, v_star = \
            MOM_LINK_ASSEMBLER_FOU(dx=dx, dy=dy,
                                   del_x=del_x, del_y=del_y,
                                   u=u, v=v,
                                   u_e=u_e, v_n=v_n,
                                   p=p_star,
                                   u_BC=u_BC, v_BC=v_BC,
                                   BC=BC,
                                   max_iter=max_mom_iter, al_mom=al_mom,
                                   rho=rho, mu=mu, sigma = sigma,
                                   dt = dt, g = g,
                                   phi = phi_CC)

        # Rhie-Chow Interpolation
        de_avg, u_e, dn_avg, v_n = \
            MOMENTUM_INTERPOLATION(dx=dx, dy=dy,
                                   uP_hat=uP_hat, vP_hat=vP_hat,
                                   d_u=d_u, d_v=d_v,
                                   p=p_star)

        # Pressure poisson solving
        p_prime, b_mass_res = \
            PPE_SOLVER(dx=dx, dy=dy,
                       u_e=u_e, v_n=v_n,
                       de_avg=de_avg, dn_avg=dn_avg,
                       u_BC=u_BC, v_BC=v_BC,
                       p_prime=p_prime, max_iter=max_p_iter)

        u_e, v_n = \
            FACE_VEL_CORRECTION(u_e=u_e, de_avg=de_avg,
                                v_n=v_n, dn_avg=dn_avg,
                                p_prime=p_prime)

        u_star, v_star = \
            CC_VEL_CORRECTION(dx=dx, dy=dy,
                              u_star=u_star, v_star=v_star,
                              d_u=d_u, d_v=d_v,
                              p_prime=p_prime, BC=BC
                              )

        p_star = p_star + al_p * p_prime
        if b_mass_res < eps_mass_tol:
            break

    u = u_star
    v = v_star
    p = p_star
    # end Solve for velocities using SIMPLE

    # Start LS advection---------------------------------------------------------
    phi_CC = LSA.LS_RK4_WENO5(phi_0 = phi_CC, u = u, v = v,
                          dt = dt, dx = dx[0,0], dy = dy[0,0])
    # End LS advection-----------------------------------------------------------

    # Start Reinitialization-----------------------------------------------------
    reinit_iter = 0
    while True:
        reinit_iter+=1
        phi_CC_old = phi_CC
        phi_CC = LSA.TVD_RK3_WENO5(phi0 = phi_CC_old,
                                   dt = dt, dx = dx[0,0], dy = dy[0,0])
        
        if np.linalg.norm(phi_CC - phi_CC_old) < 1e-4 or reinit_iter > 10:
            break
    # End Reinitialization---------------------------------------------------------


    if round(t_step % (T_max/(20*dt)),4) == 0.0:
        print("\n",f"mass res @ {t} sec = ", b_mass_res, "\n")
        
        # post.invest_traces(phi = phi_CC, p = p,
        #                    X_c = X_c, Y_c = Y_c,
        #                    dx = dx, dy = dy,
        #                    sigma = sigma,
        #                    title = (f"t = {t} sec, rho2={rho_2}, rho1={rho_1} mu2={mu_2}, mu1={mu_1}"),
        #                    V0 = V0,
        #                    L_x = L_x, L_y = L_y)

        # post.show_quiver(phi = phi_CC, X_c=X_c, Y_c=Y_c,
        #                  u = u, v = v, L_x=L_x, L_y = L_y,
        #                  title = (f"t={round(t,5)}sec"),
        #                  V0 = V0)
        
        post.PLOT_VORT_VOL_N_PRESS(phi_CC, X_c, Y_c, 
                                   u, v, L_x, L_y, p,
                                   f"t={round(t,5)}sec", V0)
        
    t_step += 1

post.invest_traces(phi=phi_CC, p=p,
                   X_c=X_c, Y_c=Y_c,
                   dx=dx, dy=dy,
                   sigma=sigma,
                   title=(f"t = {t} sec, rho2={rho_2}, rho1={rho_1} mu2={mu_2}, mu1={mu_1}"),
                   V0=V0,
                   L_x=L_x, L_y=L_y)

post.show_quiver(phi=phi_CC, X_c=X_c, Y_c=Y_c,
                 u=u, v=v, L_x=L_x, L_y=L_y,
                 title=(f"t={round(t, 5)}sec"),
                 V0=V0)
