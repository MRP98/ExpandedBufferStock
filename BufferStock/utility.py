from numba import njit

@njit(fastmath=True)
def func(c,par):
    return c**(1-par.rho)/(1-par.rho)