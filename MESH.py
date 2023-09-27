# File for mesh generation

# Predefined libraries
import numpy as np


# Predefined libraries


def CELL_CENT_UNIFORM(start: float = 0.0,
            end: float = 1.0,
            n_pts: int = 10) \
        -> (np.array, np.array, np.array):
    '''
    :param start: float, optional
        start of domain. Default => 0.0
    :param end: float, optional
        end of domain in x. Default => 1.0
    :param n_pts: int, optional
        number of cells in x direction. Default => 10
    :return dvec : span of one cell
            del_vec : distance between two consecutive cell centers
            vec_c : location of cell center
    '''

    # Making uniformly spaced grid
    zeta = np.linspace(start, end, n_pts + 1)
    dvec = zeta[1:] - zeta[:-1]
    vec_c = 0.5 * (zeta[1:] + zeta[:-1])
    del_vec = vec_c[1:] - vec_c[:-1]

    return dvec, del_vec, vec_c


def VERTEX_CENT_UNIFORM(start: float = 0.0,
                        end: float = 1.0,
                        n_pts: int = 10) \
        -> np.array:
    '''
    :param start: float, optional
        start of domain. Default => 0.0
    :param end: float, optional
        end of domain in x. Default => 1.0
    :param n_pts: int, optional
        number of cells in x direction. Default => 10
    :return del_vec : distance between two consecutive points
            vec_c : location of vertex
    '''
    vec_c = np.linspace(start, end, n_pts, endpoint=True)
    return vec_c
