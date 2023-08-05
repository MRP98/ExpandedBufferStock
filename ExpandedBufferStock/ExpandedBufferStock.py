import numpy as np
import time

from consav import ModelClass, jit
from consav.grids import nonlinspace
from consav.quadrature import create_PT_shocks

import nvfi
import last_period
import post_decision

class ExpandedBufferStockClass(ModelClass):

    def settings(self):
        pass

    def setup(self):
        ''' Define model parameters '''

        par = self.par

        par.T = 10
        par.tol = 1e-8

        # Preferences
        par.beta = 0.9
        par.rho = 3.0
        
        # Returns and income
        par.r_a = -0.015
        par.Gamma = 1.02
        par.sigma_psi = 1e-8
        par.sigma_xi = 1e-8
        par.Npsi = 5
        par.Nxi = 5
        par.mu = 0.3
        par.u = 0.07

        # Debt
        par.phi = 2*1e-8
        par.eta = 0.0
        par.r_d = 0.120
        par.lambdaa = 0.03

        # Grid density
        par.N_nb = 50
        par.N_db = 50
        par.N_n = 75    # Post-decision grid density
        par.N_d = 75    # Post-decision grid density
        
        # Grid span
        par.nb_max = 5.0
        par.nb_min = -par.phi
        par.db_max =  par.phi
        par.db_min = 1e-8

    def allocate(self):
        ''' Create grids and solution arrays '''

        par = self.par
        sol = self.sol

        # A. Pre-decision grids
        par.grid_nb = nonlinspace(par.nb_min,par.nb_max,par.N_nb,1.1)
        par.grid_db = nonlinspace(par.db_min,par.db_max,par.N_db,1.1)

        # B. Post-decision grids (dense)
        par.grid_n = nonlinspace(par.nb_min,par.nb_max,par.N_n,1.1)
        par.grid_d = nonlinspace(par.db_min,par.db_max,par.N_d,1.1)

        # C. Shocks
        shocks = create_PT_shocks(par.sigma_psi,par.Npsi,
                                  par.sigma_xi,par.Nxi,
                                  par.u,par.mu)
        
        par.psi,par.psi_w,par.xi,par.xi_w,par.Nshocks = shocks

        # D. Solution arrays
        keep_shape = (par.T,par.N_nb,par.N_db)
        post_shape = (par.T,par.N_n,par.N_d)
        sol.c_keep = np.zeros(keep_shape)
        sol.v_keep = np.zeros(keep_shape)
        sol.w = np.zeros(post_shape)

        # E. Speed
        par.time_total = np.nan
        par.time_w = np.zeros(par.T)
        par.time_adj = np.zeros(par.T)
        par.time_keep = np.zeros(par.T)

    def solve(self,do_print=True):
        ''' Solve the model '''

        tic = time.time()

        # Backwards induction
        for t in reversed(range(self.par.T)):

            if do_print == True: print(f'Period {t+1}:')

            with jit(self) as model:

                par = model.par
                sol = model.sol

                if t == par.T-1: 
                    
                    # Last period
                    tic_ = time.time()
                    last_period.solve(sol,par)
                    toc_ = time.time()

                    par.time_keep[t] = toc_-tic_
                    if do_print == True: print(f"Keeper's problem solved in {toc_-tic_:.1f} secs \n")

                else:

                    # A. Post-decision value function
                    tic_ = time.time()
                    post_decision.compute_w(t,sol,par)
                    toc_ = time.time()

                    par.time_w[t] = toc_-tic_
                    if do_print == True: print(f'Post-decision computed in {toc_-tic_:.1f} secs')

                    # B. Keeper's problem
                    tic_ = time.time()
                    nvfi.solve_keeper(t,sol,par)
                    toc_ = time.time()

                    par.time_keep[t] = toc_-tic_
                    if do_print == True: print(f"Keeper's problem solved in {toc_-tic_:.1f} secs \n")

        toc = time.time()
        self.par.time_total = toc - tic
        if do_print == True: print(f'Model solved with NVFI in {toc-tic:.1f} secs')