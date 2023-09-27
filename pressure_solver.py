import numpy as np
from numba import njit


def PPE_SOLVER(dx: np.array, dy: np.array,
               u_e: np.array, v_n: np.array,
               de_avg: np.array, dn_avg: np.array,
               u_BC: np.array, v_BC: np.array,
               p_prime: np.array, max_iter: int
               ) -> (np.array, float):
    nrows, mcols = p_prime.shape

    Fe = np.append(u_e, u_BC['right'], axis=1) * dy
    Fw = np.append(u_BC['left'], u_e, axis=1) * dy
    Fn = np.append(v_n, v_BC['top'], axis=0) * dx
    Fs = np.append(v_BC['bottom'], v_n, axis=0) * dx

    b_cont = -Fe + Fw - Fn + Fs
    ap_E, ap_W, ap_N, ap_S = np.zeros(shape=(4, nrows, mcols))

    ap_E[:, :-1] = dy * de_avg
    ap_W[:, 1:] = dy * de_avg
    ap_N[:-1, :] = dx * dn_avg
    ap_S[1:, :] = dx * dn_avg

    ap_P = ap_E + ap_W + ap_N + ap_S

    p_prime = GAUSS_SEIDEL_PR(phi=p_prime, d=1 / ap_P,
                              a_E=ap_E, a_W=ap_W,
                              a_N=ap_N, a_S=ap_S,
                              b=b_cont, max_iter=max_iter
                             )

    return p_prime, np.linalg.norm(b_cont)


@njit
def GAUSS_SEIDEL_PR(phi: np.array, d: np.array,
                    a_E: np.array, a_W: np.array,
                    a_N: np.array, a_S: np.array,
                    b: np.array  , max_iter: int
                    ) -> (np.array, np.array):
    phi_hat = np.copy(phi)
    phi_star = np.copy(phi)
    nrows, mcols = phi.shape

    for iter_p in range(max_iter):
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

    return phi_hat