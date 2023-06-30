import numpy as np

def nb_plus_func(n,d,psi,xi,par):
    ''' Transisition for n_bar '''

    cons = 1/(par.Gamma*psi)
    nb_plus = cons*((1+par.r_a)*n - (par.r_d-par.r_a)*d) + xi
    nb_plus = np.fmax(nb_plus,par.nb_min) # lower bound
    nb_plus = np.fmin(nb_plus,par.nb_max) # upper bound

    return nb_plus

def db_plus_func(d,psi,par):
    ''' Transisition for d_bar '''

    cons = 1/(par.Gamma*psi)
    db_plus = cons*(1-par.lambdaa)*d
    db_plus = np.fmax(db_plus,par.db_min) # lower bound
    db_plus = np.fmin(db_plus,par.db_max) # upper bound

    return db_plus