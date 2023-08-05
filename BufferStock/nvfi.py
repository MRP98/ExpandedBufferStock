import numpy as np
from numba import njit, prange
from consav import linear_interp
from consav import golden_section_search

import utility

@njit
def obj_bellman(c,m,interp_w,par):
    """ Evaluate Bellman equation """

    # A. End-of-period assets
    a = m-c
    
    # B. Continuation value
    w = linear_interp.interp_1d(par.grid_a,interp_w,a)

    # C. Total value
    v = utility.func(c,par) + w

    return -v # We are minimizing
    
@njit(parallel=True)
def solve_bellman(t,sol,par):
    """ Solve Bellman equation using NVFI """

    v = sol.v[t]
    c = sol.c[t]

    for i_p in prange(par.N_p):
        for i_m in range(par.N_m):
            
            # A. States
            m = par.grid_m[i_m]

            # B. Optimal choice
            c_low = np.fmin(m/2,1e-8)
            c_high = m
            c[i_p,i_m] = golden_section_search.optimizer(obj_bellman,
                                                         c_low,c_high,
                                                         args=(m,sol.w[i_p],par),
                                                         tol=par.tol)
            
            # C. Optimal value
            v[i_p,i_m] = -obj_bellman(c[i_p,i_m],m,sol.w[i_p],par)