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
        # Tangent velocity and direction follows from set v = (v_x,v_y,0.0), no rotations
        # Prescribed tangent velocity (constant), no vertical velocity
        vx_prescribed = 0.5  # m/s
        vy_prescribed = 0.3  # m/s
        vz_prescribed = 0.0  # m/s (no vertical motion)
        
        # Start location (in contact)
        z0 = R - 0.05  # Start with small overlap
        X[0,:] = np.array([x0, y0, z0])
        V[0,:] = np.array([vx_prescribed, vy_prescribed, vz_prescribed])
        
        for i in range(1, numSteps):
            # Current separation distance and cap height
            d = X[i-1,2]
            h = R - d
            
            if h <= 0:
                # No contact
                F[i,:] = np.array([0.0, 0.0, 0.0])
                T[i,:] = np.array([0.0, 0.0, 0.0])
            else:
                # Contact exists
                tau = np.arccos(d/R)
                vn = np.min([V[i-1,2], 0.0])  # Clipped normal velocity
                
                # Tangent velocity magnitude and direction
                vt_mag = np.sqrt(vx_prescribed**2 + vy_prescribed**2)
                if vt_mag > 1e-10:
                    nt = np.array([-vx_prescribed/vt_mag, -vy_prescribed/vt_mag, 0.0])
                else:
                    nt = np.array([0.0, 0.0, 0.0])
                
                # Normal force (Eq. AFn)
                Fn = np.pi*R*h*h*kn + 2*np.pi*R*h*ketan*vn
                
                # Tangent force - full partial-slip solution (Eq. ftsimp)
                # Check slip limits (Eq. sliplim)
                time_at_i = t[i]
                elastic_viscous_term = kt*vt_mag*time_at_i + ketat*vt_mag
                
                if elastic_viscous_term - mu*ketan*vn >= 0 and \
                   elastic_viscous_term - mu*ketan*vn <= mu*kn*h:
                    # Partial slip regime
                    Ft = 2*np.pi*R*h*(kt*vt_mag*time_at_i + ketat*vt_mag) - \
                         np.pi*R*(kt*vt_mag*time_at_i + ketat*vt_mag - mu*ketan*vn)**2 / (mu*kn)
                else:
                    # Full slip
                    Ft = mu*Fn
                
                # Total force
                F[i,:] = np.array([Ft*nt[0], Ft*nt[1], Fn])
                
                # Torque calculation (full partial-slip solution)
                # xi parameter for partial slip
                xi = (kt*vt_mag*time_at_i + ketat*vt_mag - mu*ketan*vn + mu*kn*d) / (mu*kn*R)
                xi = np.clip(xi, 0, 1)  # Keep in valid range [0, 1] corresponding to [0, tau]
                
                # Torque direction perpendicular to tangent velocity
                torque_dir = np.array([vy_prescribed, -vx_prescribed, 0.0])
                if vt_mag > 1e-10:
                    torque_dir = torque_dir / vt_mag
                
                # Torque magnitude (using partial slip formula)
                T_mag = 2*np.pi*R**2 * (
                    vt_mag*(kt*time_at_i + ketat)*(R/4*(1-xi**2) - d/2*(1-xi)) +
                    mu*kn/2*(R**2/3*(xi**3 - np.cos(tau)**3) - 
                             R*d*(xi**2 - np.cos(tau)**2) + 
                             d**2*(xi - np.cos(tau)))
                )
                
                T[i,:] = T_mag * torque_dir
            
            # Apply integration of motion
            A = F[i,:] / mass
            V[i,:] = V[i-1,:] + A*dt
            X[i,:] = X[i-1,:] + V[i,:]*dt + 0.5*A*dt*dt


    elif caseflag == 3: # Shear angular
        # Here we have omega = (omega_x,omega_y,0) and no linear velocities
        # Prescribed angular velocities (set high enough to reach slip condition)
        omega_x = 5.0  # rad/s
        omega_y = 3.0  # rad/s
        omega_z = 0.0
        omega_mag = np.sqrt(omega_x**2 + omega_y**2)
        
        # Start location (in contact)
        z0 = R - 0.05  # Start with small overlap
        X[0,:] = np.array([x0, y0, z0])
        V[0,:] = np.array([0.0, 0.0, 0.0])  # No linear velocity
        
        for i in range(1, numSteps):
            # Current separation distance and cap height
            d = X[i-1,2]
            h = R - d
            
            if h <= 0:
                # No contact
                F[i,:] = np.array([0.0, 0.0, 0.0])
                T[i,:] = np.array([0.0, 0.0, 0.0])
            else:
                tau = np.arccos(d/R)
                vn = np.min([V[i-1,2], 0.0])  # Clipped normal velocity
                
                # Normal force components
                # Elastic normal force
                Fn_elastic = np.pi*R*h*h*kn
                
                # Viscous normal force from bulk motion (zero here since vz=0)
                Fn_bulk_viscous = 2*np.pi*R*h*ketan*vn
                
                # Viscous normal force from rotation (Eq. fnshear) - clipping already accounted for
                Fn_rotation_viscous = ketan*R**3*omega_mag*(tau - np.sin(2*tau)/2)
                
                Fn = Fn_elastic + Fn_bulk_viscous + Fn_rotation_viscous
                
                # Check if slip condition is met (Eq. CoulombLim)
                slip_threshold = (mu*kn/ketat)*(1 - d/R)
                
                if omega_mag >= slip_threshold:
                    # Coulomb limit reached - combine viscous solution with maximum slip
                    # Viscous tangent force (Eq. for v_Omega = 0)
                    Ft_viscous_vec = np.pi*ketat*R**3*np.sin(tau)**2 * np.array([-omega_y, omega_x, 0.0])
                    
                    # Maximum slip tangent force
                    Ft_slip = mu*Fn
                    Ft_slip_vec = Ft_slip * np.array([-omega_y, omega_x, 0.0]) / omega_mag
                    
                    # Combined tangent force (viscous + slip resistance)
                    Ft_vec = Ft_viscous_vec + Ft_slip_vec
                    
                    # Maximum shear torque
                    T_mag = (np.pi*mu*kn*R**3/omega_mag) * (
                        R**2/3*(1 - np.cos(tau)**3) - 
                        R*d*(1 - np.cos(tau)**2) + 
                        d**2*(1 - np.cos(tau))
                    )
                    T_vec = T_mag * np.array([omega_y, -omega_x, 0.0]) / omega_mag
                    
                else:
                    # Below Coulomb limit - viscous solution only
                    # Viscous tangent force (Eq. for v_Omega = 0)
                    Ft_vec = np.pi*ketat*R**3*np.sin(tau)**2 * np.array([-omega_y, omega_x, 0.0])
                    
                    # Viscous torque
                    T_shear = 2*np.pi*ketat*R**3 * (
                        R*(1 - np.cos(tau)**3)/6 - d*np.sin(tau)**2/4
                    )
                    T_normal = np.pi*ketan*R**5/2 * (
                        2/3 - np.cos(tau) + np.cos(tau)**3/3
                    )
                    T_vec = (T_shear + T_normal) * np.array([omega_x, omega_y, 0.0])
                
                F[i,:] = np.array([Ft_vec[0], Ft_vec[1], Fn])
                T[i,:] = T_vec
            
            # Apply integration of motion
            A = F[i,:] / mass
            V[i,:] = V[i-1,:] + A*dt
            X[i,:] = X[i-1,:] + V[i,:]*dt + 0.5*A*dt*dt


    elif caseflag == 4: # Twist, only omega_z is non-zero
        # Solution selector: 'partial', 'stick', or 'slip'
        solution_type = 'partial'  # Hardcoded switch
        
        # Prescribed twist angular velocity
        omega_z = 10.0  # rad/s
        
        # Start location (in contact)
        z0 = R - 0.05
        X[0,:] = np.array([x0, y0, z0])
        V[0,:] = np.array([0.0, 0.0, 0.0])  # No linear velocity
        
        for i in range(1, numSteps):
            # Current separation distance and cap height
            d = X[i-1,2]
            h = R - d
            
            if h <= 0:
                # No contact
                F[i,:] = np.array([0.0, 0.0, 0.0])
                T[i,:] = np.array([0.0, 0.0, 0.0])
            else:
                tau = np.arccos(d/R)
                vn = np.min([V[i-1,2], 0.0])  # Clipped normal velocity
                time_at_i = t[i]
                
                # Normal force (same as case 1)
                Fn = np.pi*R*h*h*kn + 2*np.pi*R*h*ketan*vn
                
                # No tangent force (symmetry around z-axis)
                F[i,:] = np.array([0.0, 0.0, Fn])
                
                # Twist torque calculation
                if solution_type == 'stick':
                    # Full stick limit (theta_s → tau)
                    T_twist = -2*np.pi*R**4*(kt*omega_z*time_at_i + ketat*omega_z) * (
                        2/3 - np.cos(tau) + np.cos(tau)**3/3
                    )
                    
                elif solution_type == 'slip':
                    # Full slip limit (theta_s → 0, asymptotic for t → ∞)
                    T_twist = -2*np.pi*mu*R**3 * (
                        kn*(R*np.sin(tau)**3/3 - d*(tau/2 - np.sin(2*tau)/4)) +
                        ketan*vn*(tau/2 - np.sin(2*tau)/4)
                    )
                    
                else:  # solution_type == 'partial'
                    # Full partial-slip solution with theta_s
                    # Calculate theta_s from the slip boundary condition
                    kt_eta_term = kt*omega_z*time_at_i + ketat*omega_z
                    kn_eta_term = kn*d - ketan*vn
                    
                    # Check validity condition
                    validity_check = kt_eta_term**2 - (mu**2/R**2)*(kn_eta_term**2 - (kn*R)**2)
                    
                    if validity_check >= 0:
                        # Partial slip solution is valid
                        # Calculate theta_s
                        numerator = mu*kn_eta_term
                        denominator = R*np.sqrt((mu*kn)**2 + kt_eta_term**2)
                        atan_term = np.arctan2(kt_eta_term, mu*kn)
                        
                        theta_s = np.arccos(numerator/denominator) - atan_term
                        
                        # Ensure theta_s is in valid range [0, tau]
                        theta_s = np.clip(theta_s, 0, tau)
                        
                        # Full partial-slip torque
                        T_twist = -2*np.pi*R**3 * (
                            R*kt_eta_term*(2/3 - np.cos(theta_s) + np.cos(theta_s)**3/3) +
                            mu*kn*(R/3*(np.sin(tau)**3 - np.sin(theta_s)**3) - 
                                   d*(tau - theta_s)/2 + d*(np.sin(2*tau) - np.sin(2*theta_s))/4) +
                            mu*ketan*vn*((tau - theta_s)/2 - (np.sin(2*tau) - np.sin(2*theta_s))/4)
                        )
                    else:
                        # Outside valid range, use stick or slip as appropriate
                        if time_at_i < 0.01:  # Early time, use stick
                            T_twist = -2*np.pi*R**4*kt_eta_term * (
                                2/3 - np.cos(tau) + np.cos(tau)**3/3
                            )
                        else:  # Late time, approaching slip
                            T_twist = -2*np.pi*mu*R**3 * (
                                kn*(R*np.sin(tau)**3/3 - d*(tau/2 - np.sin(2*tau)/4)) +
                                ketan*vn*(tau/2 - np.sin(2*tau)/4)
                            )
                
                T[i,:] = np.array([0.0, 0.0, T_twist])
            
            # Apply integration of motion
            A = F[i,:] / mass
            V[i,:] = V[i-1,:] + A*dt
            X[i,:] = X[i-1,:] + V[i,:]*dt + 0.5*A*dt*dt


    elif caseflag == 5: # Roll
        # Pure rolling where v_Omega = R*(-omega_y, omega_x, 0)
        omega_x = 2.0  # rad/s
        omega_y = 1.5  # rad/s
        omega_mag = np.sqrt(omega_x**2 + omega_y**2)
        
        # Linear velocity for pure rolling
        vx_roll = -R*omega_y
        vy_roll = R*omega_x
        vz_roll = 0.0  # No vertical velocity
        
        # Start location (in contact)
        z0 = R - 0.05
        X[0,:] = np.array([x0, y0, z0])
        V[0,:] = np.array([vx_roll, vy_roll, vz_roll])
        
        for i in range(1, numSteps):
            # Current separation distance and cap height
            d = X[i-1,2]
            h = R - d
            
            if h <= 0:
                # No contact
                F[i,:] = np.array([0.0, 0.0, 0.0])
                T[i,:] = np.array([0.0, 0.0, 0.0])
            else:
                tau = np.arccos(d/R)
                vn = np.min([V[i-1,2], 0.0])  # Clipped normal velocity
                
                # Normal force components
                # Elastic normal force
                Fn_elastic = np.pi*R*h*h*kn
                
                # Viscous normal force from bulk motion (zero here since vz=0)
                Fn_bulk_viscous = 2*np.pi*R*h*ketan*vn
                
                # Viscous normal force from rotation (Eq. fnshear) - clipping already accounted for
                Fn_rotation_viscous = ketan*R**3*omega_mag*(tau - np.sin(2*tau)/2)
                
                Fn = Fn_elastic + Fn_bulk_viscous + Fn_rotation_viscous
                
                # Tangent force for roll (Eq. ftroll)
                Ft_vec = np.pi*ketat*R*h**2 * np.array([omega_y, -omega_x, 0.0])
                
                F[i,:] = np.array([Ft_vec[0], Ft_vec[1], Fn])
                
                # Roll torque (viscous)
                T_tangent = 2*np.pi*ketat*R**3 * (
                    R*(1 - 3*np.cos(tau)**2 + 2*np.cos(tau)**3)/12 -
                    d*(1 - np.cos(tau))**2/4
                )
                T_normal = np.pi*ketan*R**4/2 * (
                    2/3 - np.cos(tau) + np.cos(tau)**3/3
                )
                T_vec = -(T_tangent + T_normal) * np.array([omega_x, omega_y, 0.0])
                
                T[i,:] = T_vec
            
            # Apply integration of motion
            A = F[i,:] / mass
            V[i,:] = V[i-1,:] + A*dt
            X[i,:] = X[i-1,:] + V[i,:]*dt + 0.5*A*dt*dt
                
    else:
        print("ERROR: INVALID CASEFLAG.")

    # Merge into one array
    results = np.concatenate((X,V,F,T),axis=1)

    return results

# End of file