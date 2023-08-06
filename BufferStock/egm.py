import numpy as np
from numba import njit, prange
from consav import linear_interp

import utility

@njit(parallel=True)
def solve_bellman(t,sol,par):
    """ Solve Bellman equation using EGM """

    c = sol.c[t]

    for i_p in prange(par.N_p):

        # A. Temporary container
        m_temp = np.zeros(par.N_a+1) # m_temp[0] = 0
        c_temp = np.zeros(par.N_a+1) # c_temp[0] = 0

        # B. Use inverted Euler
        for i_a in range(par.N_a):

            # I. Find consumption
            c_temp[i_a+1] = utility.inv_marg_func(sol.q[i_p,i_a],par)

            # II. Find endogenous state 
            m_temp[i_a+1] = par.grid_a[i_a] + c_temp[i_a+1]
        
        # C. Interpolate to get consumption function
        for i_m in range(par.N_m):
            c[i_p,i_m] = linear_interp.interp_1d(m_temp,c_temp,par.grid_m[i_m])