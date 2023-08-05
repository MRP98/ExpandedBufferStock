import numpy as np
from numba import njit

@njit(fastmath=True)
def nb_plus_func(n,d,psi,xi,par):
    ''' Transition for n_bar '''

    cons = 1/(par.Gamma)
    nb_plus = cons*((1+par.r_a)*n - (par.r_d-par.r_a)*d)
    nb_plus = np.fmax(nb_plus,par.nb_min) # lower bound
    nb_plus = np.fmin(nb_plus,par.nb_max) # upper bound

    return nb_plus

@njit(fastmath=True)
def db_plus_func(d,psi,par):
    ''' Transition for d_bar '''

    cons = 1/(par.Gamma)
    db_plus = cons*(1-par.lambdaa)*d
    db_plus = np.fmax(db_plus,par.db_min) # lower bound
    db_plus = np.fmin(db_plus,par.db_max) # upper bound

    return db_plus
    