#!/usr/bin/env python3

import numpy as np

def analytical_solution(t,caseflag):
    """
    Analytical solution.
    """
    # Manual input, constants, starting pos
    R = 1.0                         # Radius sphere
    density = 2500                  # kg/m3
    g = np.array([0.0,0.0,-9.81])  # gravitational acceleration, m/s2
    mass = 4.0/3.0*np.pi*R*R*R * density    # kg

    kn = 1.0e7    # 1/s2/m2
    keta = 0.0    # 
    kt = 0.0      # 1/s2/m2

    x0 = 0.0    # m
    y0 = 0.0    # m
    z0 = 1.5    # m
    
    # Time range details
    dt = t[1] - t[0]
    numSteps = len(t)

    # Pre-allocate data
    X = np.zeros((numSteps,3))
    V = np.zeros((numSteps,3))
    F = np.zeros((numSteps,3))
    T = np.zeros((numSteps,3))

    # Start location
    X[0,:] = np.array([x0,y0,z0]) 

    # Time stepping
    for i in range(1,numSteps):
        if X[i-1,2] > R:
            # No contact
            F[i,:] = np.array([0.0,0.0,0.0])
        else:
            # Contact
            h = R - X[i-1,2]  # Cap height [m]
            Acap = 2*np.pi*R*h
            # Normal component
            fz = kn*np.pi*R*h*h - keta*V[i-1,2]*Acap


            F[i,:] = np.array([0.0,0.0,fz])

        # Apply integration of motion
        A = g + F[i,:] / mass
        V[i,:] = V[i-1,:] + A*dt
        X[i,:] = X[i-1,:] + V[i,:]*dt + 0.5*A*dt*dt


    # Merge into one array
    results = np.concatenate((X,V,F,T),axis=1)

    return results

# End of file