'''
@author: Himanshu

For LS advection
'''
import numpy as np
import ENO_funcs as ENO_F
from numpy import maximum as MAX
from numpy import minimum as MIN
from LS_funcs import Smooth_Sign as s

def LS_RK4_WENO5(phi_0: np.array,
                 u: np.array, v: np.array,
                 dt: float, dx: float, dy: float,
                 ) -> np.array:

    # Advect LS using RK-4

    # Zeroth step
    D_m_x, D_p_x, D_m_y, D_p_y = ENO_F.grad_phi(phi=phi_0, dx=dx, dy=dy)

    H_phi_0 = -(  MAX(u, 0.0) * D_m_x + MIN(u, 0.0) * D_p_x\
                + MAX(v, 0.0) * D_m_y + MIN(v, 0.0) * D_p_y)

    phi_1 = phi_0 + 0.5 * dt * H_phi_0
    # Zeroth step

    # First step
    D_m_x, D_p_x, D_m_y, D_p_y = ENO_F.grad_phi(phi=phi_1, dx=dx, dy=dy)

    H_phi_1 = -(MAX(u, 0.0) * D_m_x + MIN(u, 0.0) * D_p_x \
              + MAX(v, 0.0) * D_m_y + MIN(v, 0.0) * D_p_y)

    phi_2 = phi_1 + 0.5 * dt * (-H_phi_0 + H_phi_1)
    # First step

    # Second step
    D_m_x, D_p_x, D_m_y, D_p_y = ENO_F.grad_phi(phi=phi_2, dx=dx, dy=dy)

    H_phi_2 = -(MAX(u, 0.0) * D_m_x + MIN(u, 0.0) * D_p_x \
              + MAX(v, 0.0) * D_m_y + MIN(v, 0.0) * D_p_y)

    phi_3 = phi_2 + 0.5 * dt * (-H_phi_1 + 2*H_phi_2)
    # Second step

    # Third step
    D_m_x, D_p_x, D_m_y, D_p_y = ENO_F.grad_phi(phi=phi_3, dx=dx, dy=dy)

    H_phi_3 = -(MAX(u, 0.0) * D_m_x + MIN(u, 0.0) * D_p_x \
              + MAX(v, 0.0) * D_m_y + MIN(v, 0.0) * D_p_y)

    phi = phi_3 + (dt/6) * (H_phi_0 + 2*H_phi_1 - 4 * H_phi_2 + H_phi_3)
    # Third step

    return phi


# This subroutine is for reinitialization
def TVD_RK3_WENO5(phi0: np.array,
                  dt: float, dx: float, dy: float
                  ) -> np.array:
    
    phi = np.zeros_like(phi0)
    
    # Zeroth step
    D_m_x, D_p_x, D_m_y, D_p_y = ENO_F.grad_phi(phi=phi0, dx=dx, dy=dy)
    
    H_G_phi0 = ENO_F.H_G_hat(s(phi0, eps=dx), phi_0 = phi0,
                             u_P = D_p_x, u_M = D_m_x,
                             v_P = D_p_y, v_M = D_m_y)
    
    phi1 = phi0 + dt*(-H_G_phi0)
    # Zeroth step
    
    # First step
    D_m_x, D_p_x, D_m_y, D_p_y = ENO_F.grad_phi(phi=phi1, dx=dx, dy=dy)
    
    H_G_phi1 = ENO_F.H_G_hat(s(phi1, eps=dx), phi_0 = phi1,
                             u_P = D_p_x, u_M = D_m_x,
                             v_P = D_p_y, v_M = D_m_y)
    
    phi2 = phi1 + (dt/4)*(-3*(-H_G_phi0) + (-H_G_phi1))
    # First step
    
    # Second step
    D_m_x, D_p_x, D_m_y, D_p_y = ENO_F.grad_phi(phi=phi2, dx=dx, dy=dy)
    
    H_G_phi2 = ENO_F.H_G_hat(s(phi2, eps=dx), phi_0 = phi2,
                             u_P = D_p_x, u_M = D_m_x,
                             v_P = D_p_y, v_M = D_m_y)
    
    phi = phi2 + (dt/12)*(-(-H_G_phi0) - (-H_G_phi1) + 8*(-H_G_phi2))
    # Second step
    
    return phi