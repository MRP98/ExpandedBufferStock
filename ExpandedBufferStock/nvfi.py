import numpy as np

from numba import njit, prange
from consav import linear_interp
from consav import golden_section_search

import utility

##########
# Keeper #
##########

@njit
def obj_keep(c,nb,grid_n,grid_w,rho):
    ''' Value of choice for keeper '''

    # End-of-period net assets
    n = nb - c

    # Continuation value
    w = linear_interp.interp_1d(grid_n,grid_w,n)

    # Value of choice
    v = utility.func(c,rho) + w

    return -v

@njit(parallel=True)
def solve_keeper(t,sol,par):
    ''' Solve Bellman for keeper '''

    c = sol.c_keep[t]
    v = sol.v_keep[t]
    w = sol.w[t]

    # Loop over states
    for i_nb in prange(par.N_nb):
        for i_db in range(par.N_db):

                nb = par.grid_nb[i_nb]
                db = par.grid_db[i_db]

                # i. Optimal consumption
                c_low = 1e-8
                c_high = nb + db
                c[i_nb,i_db] = golden_section_search.optimizer(obj_keep,
                                                               c_low,
                                                               c_high,
                                                               args=(nb,par.grid_n,w[:,i_db],par.rho),
                                                               tol=par.tol)

                # ii. Optimal value    
                v[i_nb,i_db] = -obj_keep(c[i_nb,i_db],nb,par.grid_n,w[:,i_db],par.rho)
