import numpy as np

from consav import linear_interp

import utility
import trans

def compute_w(t,sol,par):
    ''' Compute post-decision value function '''

    w = sol.w[t]
    v = sol.v_keep

    # A. Loop over net assets
    for i_n in range(par.N_n):

        # B. Loop over debt level
        for i_d in range(par.N_d):

                n = par.grid_n[i_n]
                d = par.grid_d[i_d]

                # i. Loop over shocks
                for i_s in range(par.Nshocks):

                    psi_w = par.psi_w[i_s]
                    xi_w = par.xi_w[i_s]
                    psi = par.psi[i_s]
                    xi = par.xi[i_s]

                    nb_plus = trans.nb_plus_func(n,d,psi,xi,par)
                    db_plus = trans.db_plus_func(d,psi,par)

                    w[i_n,i_d] += par.beta*psi_w*xi_w*linear_interp.interp_2d(par.grid_n,
                                                                              par.grid_d,
                                                                              v[t+1],
                                                                              nb_plus,
                                                                              db_plus)




