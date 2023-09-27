import numpy as np


def FACE_VEL_CORRECTION(u_e: np.array, de_avg: np.array,
                        v_n: np.array, dn_avg: np.array,
                        p_prime: np.array) -> (np.array, np.array):
    u_e = u_e + de_avg * (p_prime[:, :-1] - p_prime[:, 1:])
    v_n = v_n + dn_avg * (p_prime[:-1, :] - p_prime[1:, :])
    return u_e, v_n


def CC_VEL_CORRECTION(dx: np.array, dy: np.array,
                      u_star: np.array, v_star: np.array,
                      d_u: np.array, d_v: np.array,
                      p_prime: np.array, BC: dict
                      ) -> (np.array, np.array):
    u = np.zeros_like(u_star)
    v = np.zeros_like(v_star)

    '''
        # Interior cells
        u[:,1:-1] = u_star[:,1:-1] + 0.5 * dy * d_u[:,1:-1] * (p_prime[:,:-2] - p_prime[:,2:])
        v[1:-1,:] = v_star[1:-1,:] + 0.5 * dx * d_v[1:-1,:] * (p_prime[:-2,:] - p_prime[2:,:])

        # East boundary cells as wall
        u[:,-1] = u_star[:,-1] + 0.5 * dy[:,0] * d_u[:,-1] *(p_prime[:,-2] - p_prime[:,-1])

        # East boundary cells as outflow --> p_e = 0.0
        u[:,-1] = u_star[:,-1] + 0.5 * dy[:,0] * d_u[:,-1] *(p_prime[:,-2] + p_prime[:,-1])

        # West boundary cells
        u[:,0] = u_star[:,0] + 0.5 * dy[:,0] * d_u[:,0] * (p_prime[:,0] - p_prime[:,1])

        # North boundary cells
        v[-1,:] = v_star[-1,:] + 0.5 * dx[0,:] * d_v[-1,:] * (p_prime[-2,:] - p_prime[-1,:])

        # South boundary cells
        v[0,:] = v_star[0,:] + 0.5 * dx[0,:] * d_v[0,:] * (p_prime[0,:] - p_prime[1,:])
    '''

    p_prime_x = 0.5 * (p_prime[:,:-1] + p_prime[:,1:])
    p_prime_y = 0.5 * (p_prime[:-1,:] + p_prime[1:,:])

    if BC['left'] == 'inlet':
        p_prime_x = np.append(p_prime[:, 0].reshape((dy.size, 1)), p_prime_x, axis=1)

    if BC['left'] == 'No_slip_wall':
        p_prime_x = np.append(p_prime[:, 0].reshape((dy.size, 1)), p_prime_x, axis=1)

    if BC['right'] == 'outflow':
        p_prime_x = np.append(p_prime_x, p_prime[:, -1].reshape((dy.size, 1)), axis=1)
        # p_prime_x = np.append(p_prime_x, np.zeros(shape = (dy.size, 1)), axis=1)

    if BC['right'] == 'No_slip_wall':
        p_prime_x = np.append(p_prime_x, p_prime[:, -1].reshape((dy.size, 1)), axis=1)

    if BC['top'] == 'No_slip_wall':
        p_prime_y = np.append(p_prime_y, p_prime[-1, :].reshape((1,dx.size)), axis=0)

    if BC['bottom'] == 'No_slip_wall':
        p_prime_y = np.append(p_prime[0, :].reshape((1,dx.size)), p_prime_y, axis=0)

    u = u_star + 0.5 * dy * d_u * (p_prime_x[:, :-1] - p_prime_x[:, 1:])
    v = v_star + 0.5 * dx * d_v * (p_prime_y[:-1, :] - p_prime_y[1:, :])

    return u, v
