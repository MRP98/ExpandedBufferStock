import numpy as np
from numba import njit, prange
from consav import linear_interp

@njit(parallel=True)
def compute_w(t,sol,par):
    """ Compute the post-decision function w """

    w = sol.w

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
                w[i_p,i_a] += psi_w*xi_w*par.beta*linear_interp.interp_2d(par.grid_p,par.grid_m,sol.v[t+1],p_plus,m_plus)