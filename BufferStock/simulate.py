import numpy as np
from numba import njit, prange
from consav import linear_interp

@njit(parallel=True)
def lifecycle(sim,sol,par):
    """ Simulate full life-cycle """

    p = sim.p
    m = sim.m
    a = sim.a
    c = sim.c
    
    for t in range(par.simT):
        for i in prange(par.simN):
            
            # Start-of-period states
            if t == 0:
                p[t,i] = 1
                m[t,i] = 1
            else:
                p[t,i] = sim.psi[t,i]*p[t-1,i]
                m[t,i] = par.R*a[t-1,i] + sim.xi[t,i]*p[t,i]

            # Choices
            c[t,i] = linear_interp.interp_2d(par.grid_p,par.grid_m,sol.c[t],p[t,i],m[t,i])
            a[t,i] = m[t,i]-c[t,i]