import numpy as np
from numba import njit, prange
from consav import linear_interp
from consav import golden_section_search

import utility

@njit
def obj_bellman(c,p,m,v_plus,par):
    """ Evaluate Bellman equation """

    # A. End-of-period assets
    a = m-c
    
    # B. Continuation value
    w = 0
    
    for i_shock in range(par.N_shocks):
            
        # I. Shocks
        psi_w = par.psi_w[i_shock]
        xi_w = par.xi_w[i_shock]
        psi = par.psi[i_shock]
        xi = par.xi[i_shock]

        # II. Next-period states
        p_plus = p*psi
        y_plus = p_plus*xi
        m_plus = par.R*a + y_plus
        
        # III. Interpolation
        w += psi_w*xi_w*par.beta*linear_interp.interp_2d(par.grid_p,par.grid_m,v_plus,p_plus,m_plus)
    
    # C. Total value
    v = utility.func(c,par) + w

    return -v # We are minimizing
    
@njit(parallel=True)
def solve_bellman(t,sol,par):
    """ Solve Bellman equation using VFI """

    c = sol.c[t]
    v = sol.v[t]

    for i_p in prange(par.N_p):
        for i_m in range(par.N_m):
            
            # A. States
            p = par.grid_p[i_p]
            m = par.grid_m[i_m]

            # B. Optimal choice
            c_low = np.fmin(m/2,1e-8)
            c_high = m
            c[i_p,i_m] = golden_section_search.optimizer(obj_bellman,
                                                         c_low,c_high,
                                                         args=(p,m,sol.v[t+1],par),
                                                         tol=par.tol)

            # C. Optimal value
            v[i_p,i_m] = -obj_bellman(c[i_p,i_m],p,m,sol.v[t+1],par)