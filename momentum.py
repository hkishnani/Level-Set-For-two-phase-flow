import numpy as np
import scipy.sparse as sparse
from numba import njit
from LS_funcs import Smooth_D_Delta as DD


# FOU_CONV, P_SOURCE, B_SOURCE, CDS_DIFF, MOM_LINK_ASSEMBLER_FOU

def FOU_CONV(dx: np.array, dy: np.array,
             u_e: np.array, v_n: np.array,
             u_BC: dict, v_BC: dict,
             rho: np.array
             ) \
        -> (np.array, np.array, np.array, np.array, np.array):
    nrows = dy.size
    mcols = dx.size

    C_P, C_E, C_W, C_N, C_S = np.zeros(shape=(5, nrows, mcols))

    Fe = rho * np.append(u_e, u_BC['right'], axis=1) * dy
    Fw = rho * np.append(u_BC['left'], u_e, axis=1) * dy
    Fn = rho * np.append(v_n, v_BC['top'], axis=0) * dx
    Fs = rho * np.append(v_BC['bottom'], v_n, axis=0) * dx

    # a_E = 0 for last col because there is no \phi_E
    C_E[:, :-1] = -np.minimum(Fe[:, :-1], 0.0)
    C_P[:, :-1] = C_P[:, :-1] + np.maximum(Fe[:, :-1], 0.0)

    # a_W = 0 for 0th col because of no \phi_W
    C_W[:, 1:] = np.maximum(Fw[:, 1:], 0.0)
    C_P[:, 1:] = C_P[:, 1:] - np.minimum(Fw[:, 1:], 0.0)

    # a_N = 0 for last row because no \phi_N
    C_N[:-1, :] = -np.minimum(Fn[:-1, :], 0.0)
    C_P[:-1, :] = C_P[:-1, :] + np.maximum(Fn[:-1, :], 0.0)

    # a_S = 0 for zeroth row because no \phi_S
    C_S[1:, :] = np.maximum(Fs[1:, :], 0.0)
    C_P[1:, :] = C_P[1:, :] - np.minimum(Fs[1:, :], 0.0)

    return C_P, C_E, C_W, C_N, C_S


def CDS_DIFF(dx: np.array, dy: np.array,
             del_x: np.array, del_y: np.array,
             BC: dict, mu: np.array) \
        -> (np.array, np.array, np.array, np.array, np.array):
    # Assuming that E,W,N,S terms remain on RHS and P on LHS
    mcols = dx.size
    nrows = dy.size

    # For interior faces take advantage of broadcasting
    D_x = dy / del_x
    D_y = dx / del_y

    # Treating boundary terms
    D_e = mu * np.append(D_x, np.zeros(shape=(nrows, 1)), axis=1)
    D_w = mu * np.append(np.zeros(shape=(nrows, 1)), D_x, axis=1)
    D_n = mu * np.append(D_y, np.zeros(shape=(1, mcols)), axis=0)
    D_s = mu * np.append(np.zeros(shape=(1, mcols)), D_y, axis=0)

    D_Eu = np.copy(D_e)
    D_Ev = np.copy(D_e)
    D_Wu = np.copy(D_w)
    D_Wv = np.copy(D_w)
    D_Nu = np.copy(D_n)
    D_Nv = np.copy(D_n)
    D_Su = np.copy(D_s)
    D_Sv = np.copy(D_s)
    D_Pu = D_e + D_w + D_n + D_s
    D_Pv = D_e + D_w + D_n + D_s

    # Do nothing for Outflow

    if BC['left'] == 'inlet':
        c = mu[:, 0] * dy[:, 0] / (3 * dx[0, 0])
        # D_Eu[:, 0] = D_Eu[:, 0] + c
        # D_Pu[:, 0] = D_Pu[:, 0] + 9 * c
        D_Ev[:, 0] = D_Ev[:, 0] + c
        D_Pv[:, 0] = D_Pv[:, 0] + 9 * c

    if BC['left'] == 'No_slip_wall':
        c = mu[:, 0] * dy[:, 0] / (3 * dx[0, 0])
        # D_Eu[:, 0] = D_Eu[:, 0] + c
        # D_Pu[:, 0] = D_Pu[:, 0] + 9 * c
        D_Ev[:, 0] = D_Ev[:, 0] + c
        D_Pv[:, 0] = D_Pv[:, 0] + 9 * c

    if BC['top'] == 'No_slip_wall':
        c = mu[-1, :] * dx[0, :] / (3 * dy[-1, 0])
        D_Su[-1, :] = D_Su[-1, :] + c
        D_Pu[-1, :] = D_Pu[-1, :] + 9 * c
        # D_Sv[-1, :] = D_Sv[-1, :] + c
        # D_Pv[-1, :] = D_Pv[-1, :] + 9 * c

    if BC['bottom'] == 'No_slip_wall':
        c = mu[0, :] * dx[0, :] / (3 * dy[0, 0])
        D_Nu[0, :] = D_Nu[0, :] + c
        D_Pu[0, :] = D_Pu[0, :] + 9 * c
        # D_Nv[0, :] = D_Nv[0, :] + c
        # D_Pv[0, :] = D_Pv[0, :] + 9 * c

    if BC['right'] == 'No_slip_wall':
        c = mu[:, -1] * dy[:, 0] / (3 * dx[0, -1])
        # D_Wu[:, -1] = D_Wu[:, -1] + c
        # D_Pu[:, -1] = D_Pu[:, -1] + 9 * c
        D_Wv[:, -1] = D_Wv[:, -1] + c
        D_Pv[:, -1] = D_Pv[:, -1] + 9 * c

    D_P = np.array([D_Pu, D_Pv])
    D_E = np.array([D_Eu, D_Ev])
    D_W = np.array([D_Wu, D_Wv])
    D_N = np.array([D_Nu, D_Nv])
    D_S = np.array([D_Su, D_Sv])

    return D_P, D_E, D_W, D_N, D_S


def P_SOURCE(dx: np.array, dy: np.array,
             p: np.array, BC: dict) \
        -> (np.array, np.array):
    # For grad p terms
    p_fx = 0.5 * (p[:, :-1] + p[:, 1:])
    p_fy = 0.5 * (p[:-1, :] + p[1:, :])

    # Left boundary
    p_fx = np.append(p[:, 0].reshape((dy.size, 1)), p_fx, axis=1)

    # Right boundary as wall
    if BC['right'] == "No_slip_wall":
        p_fx = np.append(p_fx, p[:, -1].reshape((dy.size, 1)), axis=1)

    if BC['right'] == "outflow":
        p_fx = np.append(p_fx, p[:, -1].reshape((dy.size, 1)), axis=1)
        # p_fx = np.append(p_fx, np.zeros(shape=(dy.size, 1)), axis=1)

    # Bottom boundary
    p_fy = np.append(p[0, :].reshape((1, dx.size)), p_fy, axis=0)

    # Top boundary
    p_fy = np.append(p_fy, p[-1, :].reshape((1, dx.size)), axis=0)

    S_avg_u = dy * (p_fx[:, :-1] - p_fx[:, 1:])
    S_avg_v = dx * (p_fy[:-1, :] - p_fy[1:, :])

    return S_avg_u, S_avg_v


def B_SOURCE(dx: np.array, dy: np.array,
             u_BC: dict, v_BC: dict,
             BC: dict, rho: np.array, mu: np.array
             ) \
        -> (np.array, np.array):
    nrows = dy.size
    mcols = dx.size
    b_u, b_v = np.zeros((2, nrows, mcols))

    # Adding boundary source terms according to BC
    if BC['top'] == 'No_slip_wall':
        b_u[-1, :] = b_u[-1, :] \
                     + mu[-1, :] * (8.0 / 3.0) * (dx[0, :] / dy[-1, 0]) * u_BC['top'][0, :]
        b_v[-1, :] = b_v[-1, :] \
                     + mu[-1, :] * (8.0 / 3.0) * (dx[0, :] / dy[-1, 0]) * v_BC['top'][0, :]

    if BC['bottom'] == 'No_slip_wall':
        b_u[0, :] = b_u[0, :] \
                    + mu[0, :] * (8.0 / 3.0) * (dx[0, :] / dy[0, 0]) * u_BC['bottom'][0, :]
        b_v[0, :] = b_v[0, :] \
                    + mu[0, :] * (8.0 / 3.0) * (dx[0, :] / dy[0, 0]) * v_BC['bottom'][0, :]

    if BC['left'] == 'inlet':
        b_u[:, 0] = b_u[:, 0] \
                    + rho[:, 0] * u_BC['left'][:, 0] ** 2 * dy[:, 0]  # \
        # + mu[:, 0] * (8.0 / 3.0) * (dy[:, 0] / dx[0, 0]) * u_BC['left'][:, 0]
        b_v[:, 0] = b_v[:, 0] \
                    + mu[:, 0] * (8.0 / 3.0) * (dy[:, 0] / dx[0, 0]) * v_BC['left'][:, 0]

    if BC['left'] == 'No_slip_wall':
        b_u[:, 0] = b_u[:, 0] \
                    + mu[:, 0] * (8.0 / 3.0) * (dy[:, 0] / dx[0, 0]) * u_BC['left'][:, 0]
        b_v[:, 0] = b_v[:, 0] \
                    + mu[:, 0] * (8.0 / 3.0) * (dy[:, 0] / dx[0, 0]) * v_BC['left'][:, 0]

    if BC['right'] == 'No_slip_wall':
        b_u[:, -1] = b_u[:, -1] \
                     + mu[:, -1] * (8.0 / 3.0) * (dy[:, 0] / dx[0, 0]) * u_BC['right'][:, 0]
        b_v[:, -1] = b_v[:, -1] \
                     + mu[:, -1] * (8.0 / 3.0) * (dy[:, 0] / dx[0, 0]) * v_BC['right'][:, 0]

    if BC['right'] == 'outflow':
        b_u[:, -1] = b_u[:, -1] \
                     - rho[:, -1] * u_BC['right'][:, 0] ** 2 * dy[:, 0]

    return b_u, b_v


def MOM_LINK_ASSEMBLER_FOU(dx: np.array, dy: np.array,
                           del_x: np.array, del_y: np.array,
                           u: np.array, v: np.array,
                           u_e: np.array, v_n: np.array,
                           p: np.array,
                           u_BC: dict, v_BC: dict,
                           BC: dict,
                           max_iter: int, al_mom: float,
                           rho: np.array, mu: np.array, sigma:float,
                           dt: float, g: np.array, phi: np.array
                           ) -> (np.array, np.array, np.array,
                                 np.array, np.array, np.array):

    a_P0 = rho * (dx * dy) / dt
    
    ST_x, ST_y = kappa(phi, dx, dy)
    
    b_u, b_v = B_SOURCE(dx=dx, dy=dy,
                        u_BC=u_BC, v_BC=v_BC,
                        BC=BC, rho=rho, mu=mu)

    b_u = b_u + rho * g[0] * (dx*dy) + a_P0*u + (dx*dy)*sigma*ST_x
    b_v = b_v + rho * g[1] * (dx*dy) + a_P0*v + (dx*dy)*sigma*ST_y

    S_avg_u, S_avg_v = P_SOURCE(dx=dx, dy=dy, p=p, BC=BC)
    D_P, D_E, D_W, D_N, D_S = CDS_DIFF(dx=dx, dy=dy,
                                       del_x=del_x, del_y=del_y,
                                       BC=BC, mu=mu)

    D_Pu, D_Pv = D_P
    D_Eu, D_Ev = D_E
    D_Wu, D_Wv = D_W
    D_Nu, D_Nv = D_N
    D_Su, D_Sv = D_S

    C_P, C_E, C_W, C_N, C_S = FOU_CONV(dx=dx, dy=dy,
                                       u_e=u_e, v_n=v_n,
                                       u_BC=u_BC, v_BC=v_BC,
                                       rho=rho
                                       )

    a_Pu = D_Pu + C_P + a_P0
    a_Eu = D_Eu + C_E
    a_Wu = D_Wu + C_W
    a_Nu = D_Nu + C_N
    a_Su = D_Su + C_S

    a_Pv = D_Pv + C_P + a_P0
    a_Ev = D_Ev + C_E
    a_Wv = D_Wv + C_W
    a_Nv = D_Nv + C_N
    a_Sv = D_Sv + C_S

    d_u = al_mom / a_Pu
    d_v = al_mom / a_Pv

    uP_hat, u_star = GAUSS_SEIDEL_MOM(phi=u, d=d_u,
                                      a_E=a_Eu, a_W=a_Wu,
                                      a_N=a_Nu, a_S=a_Su,
                                      b=b_u, S_avg=S_avg_u,
                                      max_iter=max_iter, al=al_mom
                                      )

    vP_hat, v_star = GAUSS_SEIDEL_MOM(phi=v, d=d_v,
                                      a_E=a_Ev, a_W=a_Wv,
                                      a_N=a_Nv, a_S=a_Sv,
                                      b=b_v, S_avg=S_avg_v,
                                      max_iter=max_iter, al=al_mom
                                      )

    return d_u, uP_hat, u_star, d_v, vP_hat, v_star


@njit
def GAUSS_SEIDEL_MOM(phi: np.array, d: np.array,
                     a_E: np.array, a_W: np.array,
                     a_N: np.array, a_S: np.array,
                     b: np.array, S_avg: np.array,
                     max_iter: int, al: float,
                     ) -> (np.array, np.array):
    phi_hat = np.copy(phi)
    phi_star = np.copy(phi)
    nrows, mcols = phi.shape

    for iter_m in range(max_iter):
        for row in range(nrows):
            for col in range(mcols):
                E, W, N, S = 0.0, 0.0, 0.0, 0.0
                if col != mcols - 1:
                    E = a_E[row, col] * phi_star[row, col + 1]

                if col != 0:
                    W = a_W[row, col] * phi_star[row, col - 1]

                if row != nrows - 1:
                    N = a_N[row, col] * phi_star[row + 1, col]

                if row != 0:
                    S = a_S[row, col] * phi_star[row - 1, col]

                phi_hat[row, col] = d[row, col] * (E + W + N + S + b[row, col])
                phi_star[row, col] = phi_hat[row, col] \
                                     + d[row, col] * S_avg[row, col] \
                                     + (1 - al) * phi[row, col]

    return phi_hat, phi_star


def MOMENTUM_INTERPOLATION(dx: np.array, dy: np.array,
                           uP_hat: np.array, vP_hat: np.array,
                           d_u: np.array, d_v: np.array,
                           p: np.array) -> (np.array, np.array,
                                            np.array, np.array):
    # For u_e face velocity
    de_avg = 0.5 * (d_u[:, :-1] + d_u[:, 1:]) * dy

    ue_hat = 0.5 * (uP_hat[:, :-1] + uP_hat[:, 1:])

    grad_p_e = p[:, :-1] - p[:, 1:]

    u_e = ue_hat + de_avg * grad_p_e

    # For v_n face velocity
    dn_avg = 0.5 * (d_v[:-1, :] + d_v[1:, :]) * dx

    vn_hat = 0.5 * (vP_hat[:-1, :] + vP_hat[1:, :])

    grad_p_n = p[:-1, :] - p[1:, :]

    v_n = vn_hat + dn_avg * grad_p_n

    return de_avg, u_e, dn_avg, v_n


def kappa(phi:np.array, dx:np.array, dy:np.array) -> np.array:

    # Dealing with del_phi/del_x
    D_x = Del_phi_x(phi, dx)
    # Dealing with del_phi/del_x

    # Dealing with del_phi/del_x
    D_y = Del_phi_y(phi, dy)
    # Dealing with del_phi/del_x

    # Dealing with del_phi/del_x
    D_xx = Del_phi_x(Del_phi_x(phi, dx), dx)
    # Dealing with del_phi/del_x

    # Dealing with del_phi/del_x
    D_yy = Del_phi_y(Del_phi_y(phi, dy), dy)
    # Dealing with del_phi/del_x

    # Dealing with del_phi/del_x
    D_xy = Del_phi_x(Del_phi_y(phi, dy), dx)
    # Dealing with del_phi/del_x

    k = -np.divide((D_y**2 * D_xx - 2 * D_x * D_y * D_xy + D_x**2 * D_yy),
                   (D_x**2 + D_y**2)**1.5
                    )
    k = np.where(np.logical_and(k > -1/dx.min(), k < 1/dx.min()),
                 k,
                 0.0)
    return k * DD(phi, eps = 1.5*dx.max()) * dx.max() * D_x/(D_x**2 + D_y**2 + 1e-6)**0.5,\
           k * DD(phi, eps = 1.5*dy.max()) * dy.max() * D_y/(D_x**2 + D_y**2 + 1e-6)**0.5
    # return k * DD(phi, eps = 1.5*dx.max())

def Del_phi_x(phi:np.array, dx:np.array)->np.array:
    D_phi_x = np.zeros_like(phi)

    D_phi_x[:, 1:-1] = (phi[:,2:] - phi[:,:-2])/(dx[0,1:-1].reshape(1,-1))
    # D_phi_x[:, 0   ] = (4*phi[:, 1] - phi[:, 2] - 3*phi[:,0])/(2*dx[0,0])
    # D_phi_x[:,-1   ] = (4*phi[:,-2] - phi[:,-3] - 3*phi[:,-1])/(-2*dx[0,-1])
    D_phi_x[:, 0   ] = (phi[:,1] - phi[:,0])/dx[0,0]
    D_phi_x[:,-1   ] = (phi[:,-1] - phi[:,-2])/dx[0,-1]
    return D_phi_x

def Del_phi_y(phi:np.array, dy:np.array) -> np.array:
    D_phi_y = np.zeros_like(phi)
    
    D_phi_y[ 1:-1,:] = (phi[2:,:] - phi[:-2,:])/(dy[1:-1])
    # D_phi_y[ 0   ,:] = (4*phi[1 ,:] - phi[ 2,:] - 3*phi[ 0,:])/(2*dy[0 ,0])
    # D_phi_y[-1   ,:] = (4*phi[-2,:] - phi[-3,:] - 3*phi[-1,:])/(-2*dy[-1,0])
    D_phi_y[ 0, : ] = (phi[ 1,:] - phi[0 ,:])/dy[0,0]
    D_phi_y[-1, : ] = (phi[-1,:] - phi[-2,:])/dy[-1,0]
    return D_phi_y