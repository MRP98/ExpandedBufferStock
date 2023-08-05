from numba import njit

@njit(fastmath=True)
def func(c,rho):
    return c**(1-rho)/(1-rho)