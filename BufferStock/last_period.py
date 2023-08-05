from numba import njit, prange

import utility

@njit(parallel=True)
def solve(t,sol,par):
    """ Solve problem in the last period """

    v = sol.v[t]
    c = sol.c[t]

    for i_p in prange(par.N_p):
        for i_m in range(par.N_m):
            
            # A. States
            p = par.grid_p[i_p]
            m = par.grid_m[i_m]

            # B. Optimal choice
            c[i_p,i_m] = m

            # C. Optimal value
            v[i_p,i_m] = utility.func(c[i_p,i_m],par)