import numpy as np
from numpy import linalg as nl
from math import pi
import MESH
from numba import njit


# Defining Smooth Heaviside function
def H(x: np.array, eps: float) -> np.array:
    H = [0.0,
         lambda x: (x + eps) / (2 * eps) + np.sin(pi * x / eps) / (2 * pi),
         1.0]

    cond_list = [x < -eps,
                 abs(x) <= eps,
                 x > eps]

    return np.piecewise(x, condlist=cond_list, funclist=H)


# Defining Smooth Heaviside function


# Defining smooth Dirac Delta function
def Smooth_D_Delta(x: np.array, eps: float):
    D_Delta = [lambda phi: (0.5/eps) * (1 + np.cos(pi * phi / eps)),
               0.0]
    cond_list = [abs(x) <= eps, abs(x) > eps]

    return np.piecewise(x, condlist=cond_list, funclist=D_Delta)


# Defining smooth Dirac Delta function


# Defining smooth Sign function
def Smooth_Sign(x: np.array, eps: float) -> np.array:
    S = x / np.sqrt(x ** 2 + eps ** 2)
    # S = np.sign(x)
    return S


# Defining smooth Sign function


# Defining minmod function
def minmod(u:np.array, v:np.array,
           eps: float) -> np.array:

    condlist = [u * v > 0.0,
                u * v <= 0.0]

    choicelist = [Smooth_Sign(u, eps) * np.minimum(np.absolute(u),
                                                   np.absolute(v)),
                  0.0]

    return np.select(condlist, choicelist)
# Defining minmod function


# Let's Patch a circle on two grids
def PATCH_CIRCLE(h: float, k: float, r: float,
                 Xc: np.array, Yc: np.array,
                 Xv: np.array, Yv: np.array
                 ) -> np.array:
    Xv_grid, Yv_grid = np.meshgrid(Xv, Yv, indexing='xy')
    Xc_grid, Yc_grid = np.meshgrid(Xc, Yc, indexing='xy')

    circ = lambda x, y: np.sqrt((x - h) ** 2 + (y - k) ** 2) - r

    phi_VC = circ(Xv_grid, Yv_grid)
    phi_CC = circ(Xc_grid, Yc_grid)

    return phi_CC, phi_VC
# Let's Patch a circle on two grids


@njit
# To map Vertex centered to Cell centered using distance weighted stencil
def VC_to_CC_STENCIL(nrows_c: int, mcols_c: int,
                     phi_VC: np.array
                     ) -> np.array:
    phi_CC = np.zeros(shape=(nrows_c, mcols_c), dtype=float)

    for row in range(nrows_c):
        for col in range(mcols_c):
            phi_CC[row, col] = np.average(phi_VC[row:row + 2, col:col + 2])

    return phi_CC


# To map Vertex centered to Cell centered using distance weighted stencil

# Newton CG Method
def Newton_CG(p, grad_p, x, V):
    '''
    Parameters
    ----------
    p : Interpolated Polynomial
    grad_p : gradient of interpolation function (CD)
    x : vector of x i + y j as numpy array
    V : Volume of Cell

    Returns
    -------
    x : Converged value of (x,y) on phi = 0
    '''

    x0 = x  # storing the initial value

    for i in range(5):
        del_1 = -p(x) * (grad_p(x) / np.dot(grad_p(x), grad_p(x)))
        x_mid = x + del_1
        del_2 = (x0 - x) - (np.dot((x0 - x), grad_p(x)) \
                            / np.dot(grad_p(x), grad_p(x))
                            ) * grad_p(x)

        x = x_mid + del_2

        if nl.norm(np.array([nl.norm(del_1), nl.norm(del_2)])) < 1e-4 * V:
            break

    return x


# Newton CG Method


# To map Cell centered to Vertex centered grid using distance weighted stencil
@njit
def CC_to_VC_STENCIL(u_CC: np.array, v_CC: np.array,
                     u_BC: dict, v_BC: dict
                     ) -> (np.array, np.array):
    nrows, mcols = u_CC.shape
    u_VC = np.zeros(shape=(nrows + 1, mcols + 1), dtype=float)
    v_VC = np.zeros(shape=(nrows + 1, mcols + 1), dtype=float)

    u_VC[1:-1, 0] = u_BC['left'].flatten()
    u_VC[1:-1, -1] = u_BC['right'].flatten()
    u_VC[0, 1:-1] = u_BC['bottom'].flatten()
    u_VC[-1, 1:-1] = u_BC['top'].flatten()

    v_VC[1:-1, 0] = v_BC['left'].flatten()
    v_VC[1:-1, -1] = v_BC['right'].flatten()
    v_VC[0, 1:-1] = v_BC['bottom'].flatten()
    v_VC[-1, 1:-1] = v_BC['top'].flatten()
    # Firstly incorporating BC

    for row in range(nrows - 1):
        for col in range(mcols - 1):
            u_VC[row + 1, col + 1] = np.average(u_CC[row:row + 2, col:col + 2])
            v_VC[row + 1, col + 1] = np.average(v_CC[row:row + 2, col:col + 2])

    return u_VC, v_VC
# To map Cell centered to Vertex centered grid using distance weighted stencil
