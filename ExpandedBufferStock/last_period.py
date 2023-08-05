from numba import njit, prange
import numpy as np
import utility

@njit(parallel=True)
def solve(sol,par):
    ''' Solve keeper's problem in the last period '''

    v = sol.v_keep[par.T-1]
    c = sol.c_keep[par.T-1]

    for i_nb in prange(par.N_nb):
        for i_db in range(par.N_db):

            nb = par.grid_nb[i_nb]
            db = par.grid_db[i_db]

            c[i_nb,i_db] = np.fmax(1e-8,nb+db)
            v[i_nb,i_db] = utility.func(c[i_nb,i_db],par.rho)