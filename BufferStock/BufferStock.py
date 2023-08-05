import time
import numpy as np

from consav import ModelClass, jit
from consav.grids import nonlinspace
from consav.quadrature import create_PT_shocks

import vfi
import nvfi
import figs
import simulate
import last_period
import post_decision

class BufferStockClass(ModelClass):

    def settings(self):
        pass

    def setup(self):
        ''' Define model parameters '''

        par = self.par

        par.T = 5
        par.tol = 1e-8
        par.solmethod = 'vfi'

        # Preferences
        par.beta = 0.96
        par.rho = 2.0
        
        # Returns and income
        par.R = 1.03
        par.sigma_psi = 0.1
        par.sigma_xi = 0.1
        par.N_psi = 6
        par.N_xi = 6
        par.pi = 0.1
        par.mu = 0.5

        # Grid density
        par.N_m = 50
        par.N_p = 50
        par.N_a = 75 # Post-decision grid density

        # Grid span
        par.m_max = 20.0
        par.m_min = 1e-8
        par.p_max = 10.0
        par.p_min = 1e-8
        par.a_max = 20.0
        par.a_min = 1e-8

        # Simulation
        par.sim_seed = 1998
        par.simT = par.T
        par.simN = 1000

    def allocate(self):
        ''' Create grids and solution arrays '''

        par = self.par
        sol = self.sol
        sim = self.sim

        # A. Grids
        par.grid_m = nonlinspace(par.m_min,par.m_max,par.N_m,1.1)
        par.grid_p = nonlinspace(par.p_min,par.p_max,par.N_p,1.1)
        par.grid_a = nonlinspace(par.a_min,par.a_max,par.N_a,1.1) # Post-decision grid (dense)

        # B. Shocks
        shocks = create_PT_shocks(par.sigma_psi,par.N_psi,
                                  par.sigma_xi,par.N_xi,
                                  par.pi,par.mu)
        
        par.psi,par.psi_w,par.xi,par.xi_w,par.N_shocks = shocks

        # C. Solution arrays
        sol.w = np.nan*np.zeros((par.N_p,par.N_a))
        sol.c = np.nan*np.ones((par.T,par.N_p,par.N_m))        
        sol.v = np.nan*np.zeros((par.T,par.N_p,par.N_m))

        # E. Speed
        par.time_sim = np.nan
        par.time_solve = np.nan
        par.time = np.zeros(par.T)
        par.time_w = np.zeros(par.T)

        # F. Simulation
        sim.p = np.nan*np.zeros((par.simT,par.simN))
        sim.m = np.nan*np.zeros((par.simT,par.simN))
        sim.a = np.nan*np.zeros((par.simT,par.simN))
        sim.c = np.nan*np.zeros((par.simT,par.simN))

        sim.psi = np.ones((par.simT,par.simN))
        sim.xi = np.ones((par.simT,par.simN))

    def solve(self,do_print=True):
        ''' Solve the model '''

        tic = time.time()
        
        # Jitted functions
        with jit(self) as model:

            par = model.par
            sol = model.sol

        # Backwards induction
        for t in reversed(range(par.T)):

            tic_ = time.time()
            if do_print == True: print(f'Period {t+1}:')

            # A. Last period
            if t == par.T-1: 
                
                last_period.solve(t,sol,par)

            # B. Other periods
            else:

                # I. Post-decision value function (only for NVFI)
                if par.solmethod == 'nvfi': 
                    
                    tic__ = time.time()
                    post_decision.compute_w(t,sol,par)
                    toc__ = time.time()

                    par.time_w[t] = toc__-tic__
                    if do_print == True: print(f'Post-decision computed in {toc__-tic__:.1f} secs')

                # II. Solve Bellman equation using solmethod
                if par.solmethod == 'vfi': vfi.solve_bellman(t,sol,par)
                if par.solmethod == 'nvfi': nvfi.solve_bellman(t,sol,par)
            
            toc_ = time.time()
            par.time[t] = toc_-tic_
            if do_print == True: print(f"Solved in {toc_-tic_:.1f} secs \n")

        toc = time.time()
        self.par.time_solve = toc - tic
        if do_print == True: print(f'Model solved with {par.solmethod} in {toc-tic:.1f} secs')

    ############
    # Simulate #
    ############

    def simulate(self,do_print=True):
        """ Simulate model """

        np.random.seed(self.par.sim_seed)

        # Jitted functions
        with jit(self) as model:

            par = model.par
            sol = model.sol
            sim = model.sim

            # Draw random numbers
            I = np.random.choice(par.N_shocks,size=(par.T,par.simN),p=par.psi_w*par.xi_w)

            sim.psi[:] = par.psi[I]
            sim.xi[:] = par.xi[I]

            # Simulate
            tic = time.time()
            simulate.lifecycle(sim,sol,par)
            toc = time.time()

        self.par.time_sim = tic - toc
        if do_print == True: print(f'Model simulated in {self.par.time_sim:.1f} secs')    

    ###########
    # Figures #
    ###########

    def consumption_function(self,t=0):
        figs.consumption_function(self,t)

    def consumption_function_interact(self):
        figs.consumption_function_interact(self)

    def lifecycle(self):
        figs.lifecycle(self)        