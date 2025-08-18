# Theoretical calculation of the tangent force in node-based LSDEM tangent history


import numpy as np
import matplotlib.pyplot as plt

def force_node(delta, i, *model_params):
    fn = model_params[0]
    kt = model_params[1]
    g = model_params[2]
    L = model_params[3]
    mu = model_params[4]
    
    xi = g*i
    force = np.zeros(delta.size)

    ind_elastic = np.logical_and(xi < delta, delta < xi + mu * fn / kt)
    force[ind_elastic] = kt * (delta[ind_elastic] - xi)

    ind_sliding = np.logical_and(delta > xi + mu * fn / kt, delta < xi + L)
    force[ind_sliding] = mu * fn

    return force


def force_total(delta, N, model_params):
    force = 0.0
    for i in range(N):
        force += force_node(delta, i, *model_params)
    return force


def main():
        
    fn = 1.
    fric = 1.
    L = 1.
    kt = 10.
    gap = 1.
    params = (fn, kt, gap, L, fric)
    
    ndata = 100000 # Need high resolution to resolve load/unload
    natoms = 10000 # Large enough to be safe
    delta = np.linspace(0.0, 10*L, ndata)
    
    
    

    for g in [0.1, 0.5, 1., 3.]:
        gap = g
        params = (fn, kt, gap, L, fric)
        plt.plot(delta, force_total(delta, 10000, params), label=f'g/L = {gap}/{L}')
    plt.title(f'fn = {fn}, mu = {fric}, L = {L}, kt = {kt}')
    plt.xlabel(r'Sliding distance, $\delta$')
    plt.ylabel(r'Total tangential force, $F_t$')
    plt.legend()
    plt.show()
    plt.close('all')
    
    gap = 0.1
    for kt in [1., 10., 100., 1000.]:
        params = (fn, kt, gap, L, fric)
        plt.plot(delta, force_total(delta, natoms, params), label=f'Fn/kt = {fn}/{kt}')
    plt.title(f'fn = {fn}, mu = {fric}, L = {L}, g = {gap}')
    plt.xlabel(r'Sliding distance, $\delta$')
    plt.ylabel(r'Total tangential force, $F_t$')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()