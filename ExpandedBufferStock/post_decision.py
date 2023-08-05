import numpy as np

from numba import njit, prange
from consav import linear_interp

import utility
import trans

@njit(parallel=True)
def compute_w(t,sol,par):
    ''' Compute post-decision value function '''

    w = sol.w[t]
    v = sol.v_keep

    # Loop over states
    for i_n in prange(par.N_n):
        for i_d in range(par.N_d):

                n = par.grid_n[i_n]
                d = par.grid_d[i_d]

                nb_plus = trans.nb_plus_func(n,d,par)
                db_plus = trans.db_plus_func(d,par)

                w[i_n,i_d] = par.beta*linear_interp.interp_2d(par.grid_nb,
                                                              par.grid_db,
                                                              v[t+1],
                                                              nb_plus,
                                                              db_plus)



