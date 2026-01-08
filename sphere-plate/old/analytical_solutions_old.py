#!/usr/bin/env python3

import numpy as np

def analytical_solution(t,caseflag):
    """
    Analytical solutions.
    """
    # Manual input, constants, starting pos
    R = 1.0                         # Radius sphere
    density = 2500                  # kg/m3
    g = np.array([0.0,0.0,-9.81])   # gravitational acceleration, m/s2
    mass = 4.0/3.0*np.pi*R*R*R * density    # kg

    kn = 1.0e8    # 1/s2/m2
    kt = 0.5e8    # 1/s2/m2
    mu = 0.5
    ketan = 1.0e6  # 
    ketat = 1.0e6  # 

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

    # Prescribed V and Omega depend on the case, should be included after the caseflag statement.

    if caseflag == 1: # Normal linear
        # Start location
        X[0,:] = np.array([x0,y0,z0]) 
        # Time stepping
        for i in range(1,numSteps):
            vn = np.min([V[i-1,2],0.0])

            if X[i-1,2] > R:
                # No contact
                F[i,:] = mass*g
            else:
                # Contact
                # Normal bounce
                h = R - X[i-1,2]  # Cap height [m]
                Acap = 2*np.pi*R*h
                # Normal component
                fz = kn*np.pi*R*h*h - ketan*vn*Acap
                # Only a z force, no torques
                F[i,:] = np.array([0.0,0.0,fz]) + mass*g

            # Apply integration of motion
            A = F[i,:] / mass
            V[i,:] = V[i-1,:] + A*dt
            X[i,:] = X[i-1,:] + V[i,:]*dt + 0.5*A*dt*dt


    elif caseflag == 2: # Tangent linear
        # tangent velocity and direction follows from set v = (v_x,v_y,0.0), no rotations

        # F_t &= 2\pi Rh\left(k_t|\vec{v}_t|t + \eta_t|\vec{v}_t|\right) - \pi R \frac{\Big(k_t|\vec{v}_t|t + \eta_t|\vec{v}_t| - \mu\,\eta_{n}v_{n}\Big)^{2}}{\mu k_{n}}
        # \theta_s =\cos^{-1}\left(\dfrac{k_t |\vec{v}_t|t+\eta_t |\vec{v}_t| - \mu \eta_n v_n + \mu k_n d}{\mu k_n R}\right)

        # Torque:
        # \vec{T}_\mathrm{t} = \frac{2\pi R^{2}}{|\vec{v}_t|}\Big[ |\vec{v}_t|(k_t t+\eta_t)\Big(\frac{R}{4}(1-\xi^{2})-\frac{d}{2}(1-\xi)\Big)+
        #     \frac{\mu k_n}{2}\Big(\frac{R^{2}}{3}(\xi^{3}-\cos^{3}\tau)-Rd(\xi^{2}-\cos^{2}\tau)+d^{2}(\xi-\cos\tau)\Big)\Big] 
        #     (v_y,-v_x,0)
        # where 
        #  \xi = \frac{k_t |\vec{v}_t| t+\eta_t |\vec{v}_t|-\mu\eta_n v_n+\mu k_n d}{\mu k_n R}
    elif caseflag == 3: # Shear angular
        # Here we have omega = (omega_x,omega_y,0) and no other velocities
        # Max tangent force kicks in at: \sqrt{ \omega_x^2 + \omega_y^2} \geq \dfrac{\mu k_n}{\eta_t} \left( 1 - \dfrac{d}{R} \right)
            # we will set omega_x and omega_y sufficiently high to reach this point
        # max tangent force : F_t^{\mathrm{slip}} = \mu F_n = \mu\left(\pi R h^2 k_n + 2 \pi Rh \eta_n v_n \right) 

        # $\tau = \cos^{-1}\left(d/R\right)$, d = z
        # Viscous normal force = \eta_n R^3 \sqrt{\omega_x^2 + \omega_y^2} \left( \tau - \frac{\sin \left(2\tau\right)}{2} \right)
        # Vicsous tangent force = \vec{F}_{t,\eta}^{\vec{v}_\Omega = 0} &= \pi \eta_t R^3 \sin^2\tau \left( -\omega_y, \omega_x, 0\right)

        # tangent direction follows from omega values, is already included in vector answer/result

        # viscous torque for rotation 
        #  \vec{T}_\eta = & \Big[\,2\pi\eta_t R^3 \left( \frac{R(1-\cos^3\tau)}{6} - \frac{d\sin^2\tau}{4} \right) + \frac{\pi\eta_n R^5}{2} \left( \frac{2}{3}-\cos\tau+\frac{\cos^3\tau}{3} \right)\Big]  ( \omega_x, \omega_y, 0 )
        # maximum torque from shear contribution:
        # \max( \vec{T}_{\mathrm{roll/shear}} ) =& \frac{\pi\mu k_n R^{3}}{\sqrt{\omega_x^2 + \omega_y^2}} \Big[\frac{R^{2}}{3}\big(1-\cos^{3}\tau\big)-Rd\big(1-\cos^{2}\tau\big)+d^{2}\big(1-\cos\tau\big)\Big] (\omega_{y},-\omega_{x},0) 
    elif caseflag == 4: # Twist, only omega_z is non-zero
        # Only a z torque remains:
        # T_\mathrm{twist} =& -2\pi R^3\Big\{R\left( k_t\omega_z t+\eta_t\omega_z\right)\Big(\frac{2}{3}-\cos\theta_s+\frac{\cos^3\theta_s}{3}\Big) \\
        # &+\mu k_n\Big[\frac{R}{3}\big(\sin^3\tau-\sin^3\theta_s\big)-d\Big(\frac{\tau-\theta_s}{2}-\frac{\sin2\tau-\sin2\theta_s}{4}\Big)\Big] \\
        # & +\mu\eta_n v_n\Big(\frac{\tau-\theta_s}{2}-\frac{\sin2\tau-\sin2\theta_s}{4}\Big)\Big\}
    elif caseflag == 5: # Roll
        # the case of pure rolling where $\vec{v}_\Omega = R\left( -\omega_y,\omega_x,0\right)$
        # For pure roll, again with $\vec{v}_\Omega = R\left( -\omega_y,\omega_x,0\right)$, the torque is
        # \begin{equation}
        # \begin{aligned}
        #     \vec{T}_\eta^\mathrm{roll}=&-\Bigg[2\pi\eta_t R^3\Big(\frac{R\big(1-3\cos^2\tau+2\cos^3\tau\big)}{12}-\frac{d(1-\cos\tau)^2}{4}\Big) \\
        #     & +\frac{\pi}{2}\eta_n R^4\Big(\frac{2}{3}-\cos\tau+\frac{\cos^3\tau}{3}\Big)\Bigg]\\
        #     &(\omega_x,\omega_y,0) \text{.}
        # \end{aligned}
        # \end{equation}

        # roll tangent force \vec{F}_{t,\eta}^{\text{roll}} &= \pi \eta_t R h^2 \left( \omega_y,-\omega_x,0\right)
        # roll normal force is \eta_n R^3 \sqrt{\omega_x^2 + \omega_y^2} \left( \tau - \frac{\sin \left(2\tau\right)}{2} \right)
    else:
        print("ERROR: INVALID CASEFLAG.")

    # Merge into one array
    results = np.concatenate((X,V,F,T),axis=1)

    return results

# End of file