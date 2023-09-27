# ENO-3 jiang and Peng
"""
ENO-WENO Schemes for implementation on interior cells of 12 x 12 grid size
+X+X+X+X+X+X+X+X+X+X+X+X+X+X+X+X+X+X+X+X+X+X+X+X+X+X+X+X+X+X+X+X+X+X+X+X+X+X+X+X+X+X+X+X+X+X+X+X

       0      1      2      .      .      .      .      .      .     m-3    m-2    m-1
     _____  _____  _____  _____  _____  _____  _____  _____  _____  _____  _____  _____
    |     ||     ||     ||     ||     ||     ||     ||     ||     ||     ||     ||     |
11  |  F  ||  F  ||  F  ||  F  ||  F  ||  F  ||  F  ||  F  ||  F  ||  F  ||  F  ||  F  |  n-1
    |_____||_____||_____||_____||_____||_____||_____||_____||_____||_____||_____||_____|
    |     ||     ||     ||     ||     ||     ||     ||     ||     ||     ||     ||     |
10  |  F  ||  E  ||  E  ||  E  ||  E  ||  E  ||  E  ||  E  ||  E  ||  E  ||  E  ||  F  |  n-2
    |_____||_____||_____||_____||_____||_____||_____||_____||_____||_____||_____||_____|
    |     ||     ||     ||     ||     ||     ||     ||     ||     ||     ||     ||     |
9   |  F  ||  E  ||  E  ||  E  ||  E  ||  E  ||  E  ||  E  ||  E  ||  E  ||  E  ||  F  |  n-3
    |_____||_____||_____||_____||_____||_____||_____||_____||_____||_____||_____||_____|
    |     ||     ||     ||     ||     ||     ||     ||     ||     ||     ||     ||     |
8   |  F  ||  E  ||  E  ||  W  ||  W  ||  W  ||  W  ||  W  ||  W  ||  E  ||  E  ||  F  |   .
    |_____||_____||_____||_____||_____||_____||_____||_____||_____||_____||_____||_____|
    |     ||     ||     ||     ||     ||     ||     ||     ||     ||     ||     ||     |
7   |  F  ||  E  ||  E  ||  W  ||  W  ||  W  ||  W  ||  W  ||  W  ||  E  ||  E  ||  F  |   .
    |_____||_____||_____||_____||_____||_____||_____||_____||_____||_____||_____||_____|
    |     ||     ||     ||     ||     ||     ||     ||     ||     ||     ||     ||     |
6   |  F  ||  E  ||  E  ||  W  ||  W  ||  W  ||  W  ||  W  ||  W  ||  E  ||  E  ||  F  |   .
    |_____||_____||_____||_____||_____||_____||_____||_____||_____||_____||_____||_____|
    |     ||     ||     ||     ||     ||     ||     ||     ||     ||     ||     ||     |
5   |  F  ||  E  ||  E  ||  W  ||  W  ||  W  ||  W  ||  W  ||  W  ||  E  ||  E  ||  F  |   .
    |_____||_____||_____||_____||_____||_____||_____||_____||_____||_____||_____||_____|
    |     ||     ||     ||     ||     ||     ||     ||     ||     ||     ||     ||     |
4   |  F  ||  E  ||  E  ||  W  ||  W  ||  W  ||  W  ||  W  ||  W  ||  E  ||  E  ||  F  |   .
    |_____||_____||_____||_____||_____||_____||_____||_____||_____||_____||_____||_____|
    |     ||     ||     ||     ||     ||     ||     ||     ||     ||     ||     ||     |
3   |  F  ||  E  ||  E  ||  W  ||  W  ||  W  ||  W  ||  W  ||  W  ||  E  ||  E  ||  F  |   .
    |_____||_____||_____||_____||_____||_____||_____||_____||_____||_____||_____||_____|
    |     ||     ||     ||     ||     ||     ||     ||     ||     ||     ||     ||     |
2   |  F  ||  E  ||  E  ||  E  ||  E  ||  E  ||  E  ||  E  ||  E  ||  E  ||  E  ||  F  |   2
    |_____||_____||_____||_____||_____||_____||_____||_____||_____||_____||_____||_____|
    |     ||     ||     ||     ||     ||     ||     ||     ||     ||     ||     ||     |
1   |  F  ||  E  ||  E  ||  E  ||  E  ||  E  ||  E  ||  E  ||  E  ||  E  ||  E  ||  F  |   1
    |_____||_____||_____||_____||_____||_____||_____||_____||_____||_____||_____||_____|
    |     ||     ||     ||     ||     ||     ||     ||     ||     ||     ||     ||     |
0   |  F  ||  F  ||  F  ||  F  ||  F  ||  F  ||  F  ||  F  ||  F  ||  F  ||  F  ||  F  |   0
    |_____||_____||_____||_____||_____||_____||_____||_____||_____||_____||_____||_____|
       0      1      2      3      4      5      6      7      8      9      10     11
"""
import numpy as np

from LS_funcs import Smooth_Sign as sms

from numpy import maximum as MAX
from numpy import minimum as MIN


# Godunov Flux
def H_G_hat(s: np.array, phi_0: np.array,
            u_P: np.array, u_M: np.array,
            v_P: np.array, v_M: np.array):
    '''

    :param s: Sign array for phi_0
    :param phi_0: variable
    :param u_P: phi_x+
    :param u_M: phi_x-
    :param v_P: phi_y+
    :param v_M: phi_x-
    :return: Godunov Flux
    '''

    choice_list = \
        [
            s * (np.sqrt(MAX(-MIN(u_P, 0.0), MAX(u_M, 0.0)) ** 2
                         +
                         MAX(-MIN(v_P, 0.0), MAX(v_M, 0.0)) ** 2
                         )
                 - 1)
            ,
            s * (np.sqrt(MAX(MAX(u_P, 0.0), -MIN(u_M, 0.0)) ** 2
                         +
                         MAX(MAX(v_P, 0.0), -MIN(v_M, 0.0)) ** 2
                         )
                 - 1)
        ]

    cond_list = [phi_0 >= 0.0,
                 phi_0 < 0.0]

    return np.select(condlist=cond_list, choicelist=choice_list)


# Godunov Flux


# Defining WENO weights
def w0(a_0: np.array, a_1: np.array, a_2: np.array) -> np.array:
    return a_0 / (a_0 + a_1 + a_2)


def w2(a_0: np.array, a_1: np.array, a_2: np.array) -> np.array:
    return a_2 / (a_0 + a_1 + a_2)


def al_0(a: np.array, b: np.array) -> np.array:
    return 1 * (1e-4 + 13 * (a - b) ** 2 + 3 * (a - 3 * b) ** 2) ** -2


def al_1(b: np.array, c: np.array) -> np.array:
    return 6 * (1e-4 + 13 * (b - c) ** 2 + 3 * (b + c) ** 2) ** -2


def al_2(c: np.array, d: np.array) -> np.array:
    return 3 * (1e-4 + 13 * (c - d) ** 2 + 3 * (3 * c - d) ** 2) ** -2


def phi_WENO(a: np.array, b: np.array,
             c: np.array, d: np.array):
    return (1.0 / 3.0) * w0(al_0(a, b), al_1(b, c), al_2(c, d)) * (a - 2 * b + c) \
        + (1.0 / 6.0) * (w2(al_0(a, b), al_1(b, c), al_2(c, d)) - 0.5) * (b - 2 * c + d)


# Defining WENO weights


# L(u, v)
def grad_phi(phi: np.array,
             dx: float, dy: float
             ) -> (np.array, np.array, np.array, np.array):

    nrows, mcols = phi.shape

    D_m_x, D_p_x = np.zeros(shape=(2, nrows, mcols), dtype=float)
    D_m_y, D_p_y = np.zeros(shape=(2, nrows, mcols), dtype=float)

    a_mx, b_mx, c_mx, d_mx = np.zeros(shape=(4, nrows, mcols), dtype=float)
    a_px, b_px, c_px, d_px = np.zeros(shape=(4, nrows, mcols), dtype=float)

    a_my, b_my, c_my, d_my = np.zeros(shape=(4, nrows, mcols), dtype=float)
    a_py, b_py, c_py, d_py = np.zeros(shape=(4, nrows, mcols), dtype=float)

    # X-advect WENO
    cs, ce = 3, mcols - 3
    a_mx[:, cs:ce] = (phi[:, cs - 1:ce - 1] - 2 * phi[:, cs - 2:ce - 2] + phi[:, cs - 3:ce - 3]) / dx
    b_mx[:, cs:ce] = (phi[:, cs:ce] - 2 * phi[:, cs - 1:ce - 1] + phi[:, cs - 2:ce - 2]) / dx
    c_mx[:, cs:ce] = (phi[:, cs + 1:ce + 1] - 2 * phi[:, cs:ce] + phi[:, cs - 1:ce - 1]) / dx
    d_mx[:, cs:ce] = (phi[:, cs + 2:ce + 2] - 2 * phi[:, cs + 1:ce + 1] + phi[:, cs:ce]) / dx

    a_px[:, cs:ce] = (phi[:, cs + 3: ce + 3] - 2 * phi[:, cs + 2:ce + 2] + phi[:, cs + 1:ce + 1]) / dx
    b_px[:, cs:ce] = (phi[:, cs + 2: ce + 2] - 2 * phi[:, cs + 1:ce + 1] + phi[:, cs:ce]) / dx
    c_px[:, cs:ce] = (phi[:, cs + 1: ce + 1] - 2 * phi[:, cs:ce] + phi[:, cs - 1:ce - 1]) / dx
    d_px[:, cs:ce] = (phi[:, cs: ce] - 2 * phi[:, cs - 1:ce - 1] + phi[:, cs - 2:ce - 2]) / dx

    D_p_x[:, cs:ce] = (1.0 / (12.0 * dx)) * (phi[:, cs - 2:ce - 2] - 8 * phi[:, cs - 1:ce - 1]
                                             - phi[:, cs + 2:ce + 2] + 8 * phi[:, cs + 1:ce + 1]
                                             ) \
                      + phi_WENO(a_px, b_px, c_px, d_px)[:, cs:ce]

    D_m_x[:, cs:ce] = (1.0 / (12.0 * dx)) * (phi[:, cs - 2:ce - 2] - 8 * phi[:, cs - 1:ce - 1]
                                             - phi[:, cs + 2:ce + 2] + 8 * phi[:, cs + 1:ce + 1]
                                             ) \
                      - phi_WENO(a_mx, b_mx, c_mx, d_mx)[:, cs:ce]

    # (col_i = 1 to cs)
    D_m_x[:, 1:cs] = (phi[:, 1:cs] - phi[:, 0:cs - 1]) / dx
    # (col_i = 1 to cs)

    # (col_i = 0 to cs)
    D_p_x[:, 0:cs] = (phi[:, 1:cs + 1] - phi[:, 0:cs]) / dx
    # (col_i = 0 to cs)

    # (col_i = ce to mcols)
    D_m_x[:, ce:mcols] = (phi[:, ce:mcols] - phi[:, ce - 1:mcols - 1]) / dx
    # (col_i = ce to mcols)

    # (col_i = ce to mcols-1)
    D_p_x[:, ce:mcols - 1] = (phi[:, ce + 1:mcols] - phi[:, ce:mcols - 1]) / dx
    # (col_i = ce to mcols-1)
    # X-advect WENO

    # Y-advect WENO
    rs, re = 3, nrows - 3
    a_my[rs:re, :] = (phi[rs - 1:re - 1, :] - 2 * phi[rs - 2:re - 2, :] + phi[rs - 3:re - 3, :]) / dy
    b_my[rs:re, :] = (phi[rs:re, :] - 2 * phi[rs - 1:re - 1, :] + phi[rs - 2:re - 2, :]) / dy
    c_my[rs:re, :] = (phi[rs + 1:re + 1, :] - 2 * phi[rs:re, :] + phi[rs - 1:re - 1, :]) / dy
    d_my[rs:re, :] = (phi[rs + 2:re + 2, :] - 2 * phi[rs + 1:re + 1, :] + phi[rs:re, :]) / dy

    a_py[rs:re, :] = (phi[rs + 3: re + 3, :] - 2 * phi[rs + 2:re + 2, :] + phi[rs + 1:re + 1, :]) / dy
    b_py[rs:re, :] = (phi[rs + 2: re + 2, :] - 2 * phi[rs + 1:re + 1, :] + phi[rs:re, :]) / dy
    c_py[rs:re, :] = (phi[rs + 1: re + 1, :] - 2 * phi[rs:re, :] + phi[rs - 1:re - 1, :]) / dy
    d_py[rs:re, :] = (phi[rs: re, :] - 2 * phi[rs - 1:re - 1, :] + phi[rs - 2:re - 2, :]) / dy

    D_p_y[rs:re, :] = (1.0 / (12.0 * dy)) * (phi[rs - 2:re - 2, :] - 8 * phi[rs - 1:re - 1, :]
                                             - phi[rs + 2:re + 2, :] + 8 * phi[rs + 1:re + 1, :]
                                             ) \
                      + phi_WENO(a_py, b_py, c_py, d_py)[rs:re,:]

    D_m_y[rs:re, :] = (1.0 / (12.0 * dy)) * (phi[rs - 2:re - 2, :] - 8 * phi[rs - 1:re - 1, :]
                                             - phi[rs + 2:re + 2, :] + 8 * phi[rs + 1:re + 1, :]
                                             ) \
                      - phi_WENO(a_my, b_my, c_my, d_my)[rs:re,:]

    # (row_j = 1 to rs)
    D_m_y[1:rs, :] = (phi[1:rs, :] - phi[0:rs - 1, :]) / dy
    # (row_i = 1 to rs)

    # (row_i = 0 to rs)
    D_p_y[0:rs, :] = (phi[1:rs + 1, :] - phi[0:rs, :]) / dy
    # (row_i = 0 to rs)

    # (row_i = re to nrows)
    D_m_y[re:nrows, :] = (phi[re:nrows, :] - phi[re - 1:nrows - 1, :]) / dy
    # (col_i = re to mcols)

    # (col_i = re to mcols-1)
    D_p_y[re:nrows - 1, :] = (phi[re + 1:nrows, :] - phi[re:nrows - 1, :]) / dy
    # (col_i = re to mcols-1)
    # Y-advect WENO

    return D_m_x, D_p_x, D_m_y, D_p_y
