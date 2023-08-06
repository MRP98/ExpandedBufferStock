import numpy as np
from numba import njit, prange
from consav import linear_interp

import utility

@njit(parallel=True)
def compute_wq(t,sol,par,compute_w=False,compute_q=False):
    """ Compute the post-decision functions w and q """

    w = sol.w
    q = sol.q

    for i_p in prange(par.N_p):
        for i_a in range(par.N_a):

            # States
            p = par.grid_p[i_p]
            a = par.grid_a[i_a]

            w[i_p,i_a] = 0

            for i_shock in range(par.N_shocks):

                # A. Shocks
                psi_w = par.psi_w[i_shock]
                xi_w = par.xi_w[i_shock]
                psi = par.psi[i_shock]
                xi = par.xi[i_shock]

                # B. Next period
                p_plus = p*psi
                y_plus = p_plus*xi
                m_plus = par.R*a + y_plus

                # C. Interpolation
                if compute_w == True: 
                    w[i_p,i_a] += psi_w*xi_w*par.beta*linear_interp.interp_2d(par.grid_p,par.grid_m,sol.v[t+1],p_plus,m_plus)
                
                if compute_q == True: 
                    c_plus_temp = linear_interp.interp_2d(par.grid_p,par.grid_m,sol.c[t+1],p_plus,m_plus)
                    q[i_p,i_a] += psi_w*xi_w*par.R*par.beta*utility.marg_func(c_plus_temp,par)