#!/usr/bin/env python3

import numpy as np


# ===========================================================================
#  Reusable analytical building blocks
#
#  This module collects the closed-form sphere-plane contact solutions of the
#  manuscript (Appendix 1) so that the LS-DEM implementation can be checked
#  against them. Each function transcribes one manuscript expression; the
#  case drivers further below assemble them into the five verification tests.
#
#  Sign / geometry convention (matching the manuscript):
#    - The plane / half-space is at z = 0, the sphere centre at z = d.
#    - Separation distance d = z, cap height h = R - d, half-angle
#      tau = arccos(d/R), strain eps = h/R = 1 - cos(tau).
#    - Normal approach velocity vn > 0 means approaching / compressing.
#    - The total normal stress is limited to be repulsive (Eq. sigmalim),
#      sigma_n <- max(0, sigma_n). The limit acts on the STRESS, not on the
#      velocity.
# ===========================================================================


def cap_geometry(d, R):
    """Cap height h and half-angle tau for separation distance d."""
    h = R - d
    tau = np.arccos(np.clip(d/R, -1.0, 1.0))
    return h, tau


def normal_force_linear(h, vn, R, kn, etan):
    """
    Repulsive normal force on a sphere in contact with a plane for a *uniform*
    (bulk) normal velocity, i.e. pure normal linear motion (Eq. AFn).

    Sign convention: vn > 0 means approaching / compressing the contact.

    The total normal stress is limited to be repulsive (Eq. sigmalim),
        sigma_n <- max(0, sigma_n) ,
    NOT the velocity. During separation the viscous term can therefore reduce
    the force; the outermost ring of the cap (where the elastic stress is
    smallest) goes tensile first and is clipped, while the interior still
    carries a reduced viscous force. This is the physically correct behaviour:
    a cohesionless contact loses load from the edge inward, it does not simply
    switch off all damping on the way out.

    Cap height h = R - d, with d the separation distance (z of the centre).
    """
    if h <= 0.0:
        # No contact
        return 0.0
    if vn >= 0.0:
        # Compression: edge stress = etan*vn >= 0, so the whole cap is
        # repulsive and no clipping occurs (Eq. AFn).
        return np.pi*R*kn*h*h + 2.0*np.pi*R*h*etan*vn
    # Separation (vn < 0): the stress goes tensile near the cap edge and is
    # clipped at sigma_n = 0. The repulsive region extends to an effective
    # cap height h_eff where sigma_n = 0.
    h_eff = h + etan*vn/kn           # h_eff = h + etan*vn/kn (vn < 0 -> h_eff < h)
    if h_eff <= 0.0:
        # Entire cap is tensile -> contact effectively lost
        return 0.0
    return np.pi*R*kn*h_eff*h_eff


def spurious_normal_force_rotation(Omega, d, R, kn, etan):
    """
    Diagnostic ONLY: the small residual repulsive normal force that survives
    the stress limit (Eq. sigmalim) for a pure spin Omega = sqrt(wx^2+wy^2).

    The corrected analytical solution drops the rotation-induced normal force
    to leading order, because the rotational normal velocity is anti-symmetric
    about the spin axis while the elastic stress is axisymmetric, so the
    unclipped viscous contribution integrates to zero. The stress limit only
    clips a thin tensile sliver near the receding cap edge, leaving this
    residual. It is NOT added to the force solution; it is provided so one can
    predict the expected analytical-vs-LAMMPS deviation in the normal force for
    the shear and roll tests.

    Per fixed theta, a(theta) = etan*R*Omega*sin(theta) is the peak viscous
    normal stress and b(theta) = kn*(R cos theta - d) the (axisymmetric)
    elastic stress. The azimuthal integral of the clipped tensile part has the
    closed form
        J(theta) = 2 sqrt(a^2 - b^2) - b*pi + 2 b*arcsin(b/a)   for a > b >= 0,
    and J = 0 otherwise. The residual is
        dFn = R^2 * integral_0^tau  J(theta) sin(theta) dtheta .

    This matches a direct 2D integration of max(0, -sigma_n) over the cap.
    Scaling: ~ Omega^2 at low spin and ~ eps^1.1..1.2 over DEM-relevant
    overlaps; always a fraction of the (incorrect) velocity-clipped value
    etan*R^3*Omega*(tau - sin(2 tau)/2).
    """
    h, tau = cap_geometry(d, R)
    if h <= 0.0:
        return 0.0
    n = 2000
    th = np.linspace(0.0, tau, n)
    a = etan*R*Omega*np.sin(th)
    b = kn*(R*np.cos(th) - d)
    mask = (a > b) & (a > 0.0)
    J = np.zeros_like(th)
    aa, bb = a[mask], b[mask]
    J[mask] = 2.0*np.sqrt(aa*aa - bb*bb) - bb*np.pi + 2.0*bb*np.arcsin(bb/aa)
    return R*R*np.trapezoid(J*np.sin(th), th)


# --------------------------- Tangent: linear velocity ----------------------
def slip_angle_linear(vt, vn, t, d, R, kn, kt, etan, etat, mu):
    """Slip boundary angle theta_s for a constant linear tangent velocity."""
    drive = kt*vt*t + etat*vt                       # kt|vt|t + etat|vt|
    arg = (drive - mu*etan*vn + mu*kn*d)/(mu*kn*R)
    return np.arccos(np.clip(arg, -1.0, 1.0))


def tangent_force_linear(h, d, vt, vn, t, R, kn, kt, etan, etat, mu):
    """
    Tangent force magnitude for a constant linear tangent velocity vt and a
    constant normal velocity vn (Eq. ftvt / ftsimp), with the partial-slip
    region between the stick (Eq. for theta_s -> tau) and full-slip
    (theta_s -> 0) limits.
    """
    drive = kt*vt*t + etat*vt                       # elastic + viscous drive
    lo = drive - mu*etan*vn                         # lower term of Eq. sliplim
    Fn = normal_force_linear(h, vn, R, kn, etan)
    if lo <= 0.0:
        # Entire cap sticks (purely elastic limit, will not realistically occur)
        return 2.0*np.pi*R*h*drive
    if lo >= mu*kn*h:
        # Entire cap slips
        return mu*Fn
    # Partial slip (Eq. ftsimp)
    return 2.0*np.pi*R*h*drive - np.pi*R*lo*lo/(mu*kn)


def torque_tangent_linear(h, d, vt, vn, t, R, kn, kt, etan, etat, mu):
    """
    Torque vector for a constant linear tangent velocity vt = (vx, vy, 0)
    (Eq. T_t), with stick and slip limits. The direction is (vy, -vx, 0)
    (built by the caller); here we return the scalar magnitude that multiplies
    that direction.
    """
    tau = np.arccos(np.clip(d/R, -1.0, 1.0))
    drive = kt*vt*t + etat*vt
    lo = drive - mu*etan*vn
    if lo <= 0.0:
        # Stick limit (theta_s -> tau)
        return 2.0*np.pi*R**2*(kt*t + etat)*(R/4.0*np.sin(tau)**2
                                             - d/2.0*(1.0 - np.cos(tau)))
    if lo >= mu*kn*h:
        # Slip limit (theta_s -> 0)
        return (2.0*np.pi*R**2/vt)*(mu*kn/2.0)*(
            R**2/3.0*(1.0 - np.cos(tau)**3)
            - R*d*(1.0 - np.cos(tau)**2)
            + d**2*(1.0 - np.cos(tau)))
    # Partial slip; xi = cos(theta_s)
    xi = (drive - mu*etan*vn + mu*kn*d)/(mu*kn*R)
    return (2.0*np.pi*R**2/vt)*(
        vt*(kt*t + etat)*(R/4.0*(1.0 - xi**2) - d/2.0*(1.0 - xi))
        + (mu*kn/2.0)*(R**2/3.0*(xi**3 - np.cos(tau)**3)
                       - R*d*(xi**2 - np.cos(tau)**2)
                       + d**2*(xi - np.cos(tau))))


# --------------------------- Shear / roll: angular velocity ----------------
def coulomb_slip_met(Omega, d, R, kn, etat, mu):
    """Maximum shear/roll condition (Eq. CoulombLim): the dashpot alone reaches
    the Coulomb limit over the whole cap."""
    return Omega >= (mu*kn/etat)*(1.0 - d/R)


def tangent_force_shear_viscous(omega_x, omega_y, tau, R, etat):
    """Purely viscous tangent force for pure shear (v_Omega = 0)."""
    return np.pi*etat*R**3*np.sin(tau)**2 * np.array([-omega_y, omega_x, 0.0])


def tangent_force_roll_viscous(omega_x, omega_y, h, R, etat):
    """Viscous rolling resistance (Eq. ftroll), opposite sign to shear."""
    return np.pi*etat*R*h*h * np.array([omega_y, -omega_x, 0.0])


def max_shear_roll_force(Fn, mu):
    """Maximum shear/roll tangent force magnitude (= mu*Fn = Ft_slip)."""
    return mu*Fn


def torque_max_shear_roll(omega_x, omega_y, d, R, kn, mu):
    """Maximum torque due to shear and/or roll at the Coulomb limit."""
    Omega = np.hypot(omega_x, omega_y)
    tau = np.arccos(np.clip(d/R, -1.0, 1.0))
    mag = (np.pi*mu*kn*R**3/Omega)*(
        R**2/3.0*(1.0 - np.cos(tau)**3)
        - R*d*(1.0 - np.cos(tau)**2)
        + d**2*(1.0 - np.cos(tau)))
    return mag*np.array([omega_y, -omega_x, 0.0])


def torque_shear_viscous(omega_x, omega_y, d, R, etan, etat):
    """Viscous torque for pure shear (T_eta)."""
    tau = np.arccos(np.clip(d/R, -1.0, 1.0))
    # NOTE: transcribed verbatim from the manuscript; the normal-viscous term
    # carries R^5 here versus R^4 in the roll expression below.
    coeff = (2.0*np.pi*etat*R**3*(R*(1.0 - np.cos(tau)**3)/6.0
                                  - d*np.sin(tau)**2/4.0)
             + np.pi*etan*R**5/2.0*(2.0/3.0 - np.cos(tau) + np.cos(tau)**3/3.0))
    return coeff*np.array([omega_x, omega_y, 0.0])


def torque_roll_viscous(omega_x, omega_y, d, R, etan, etat):
    """Viscous torque for pure roll (T_eta^roll)."""
    tau = np.arccos(np.clip(d/R, -1.0, 1.0))
    coeff = (2.0*np.pi*etat*R**3*(R*(1.0 - 3.0*np.cos(tau)**2 + 2.0*np.cos(tau)**3)/12.0
                                  - d*(1.0 - np.cos(tau))**2/4.0)
             + np.pi/2.0*etan*R**4*(2.0/3.0 - np.cos(tau) + np.cos(tau)**3/3.0))
    return -coeff*np.array([omega_x, omega_y, 0.0])


# --------------------------- Twist: omega_z only ---------------------------
def twist_torque(t, omega_z, vn, d, R, kn, kt, etan, etat, mu, mode='partial'):
    """
    Twist torque about the z axis for a constant omega_z (always a pure z
    torque). mode = 'partial' (general), 'stick', or 'slip'.
    """
    tau = np.arccos(np.clip(d/R, -1.0, 1.0))
    drive = kt*omega_z*t + etat*omega_z

    if mode == 'stick':
        # Stick limit (theta_s -> tau)
        return -2.0*np.pi*R**4*drive*(2.0/3.0 - np.cos(tau) + np.cos(tau)**3/3.0)

    if mode == 'slip':
        # Slip limit (theta_s -> 0), asymptotic for t -> inf
        return -2.0*np.pi*mu*R**3*(
            kn*(R*np.sin(tau)**3/3.0 - d*(tau/2.0 - np.sin(2.0*tau)/4.0))
            + etan*vn*(tau/2.0 - np.sin(2.0*tau)/4.0))

    # Partial slip (general). Validity condition:
    #   drive^2 >= (mu^2/R^2)((kn d - etan vn)^2 - kn^2 R^2)
    valid = drive**2 >= (mu**2/R**2)*((kn*d - etan*vn)**2 - (kn*R)**2)
    if not valid:
        # Outside the partial-slip regime; fall back to the stick limit.
        return twist_torque(t, omega_z, vn, d, R, kn, kt, etan, etat, mu, 'stick')

    # Slip angle theta_s(t)
    numerator = mu*(kn*d - etan*vn)
    denominator = R*np.sqrt((mu*kn)**2 + drive**2)
    theta_s = (np.arccos(np.clip(numerator/denominator, -1.0, 1.0))
               - np.arctan2(drive, mu*kn))
    theta_s = np.clip(theta_s, 0.0, tau)

    return -2.0*np.pi*R**3*(
        R*drive*(2.0/3.0 - np.cos(theta_s) + np.cos(theta_s)**3/3.0)
        + mu*kn*(R/3.0*(np.sin(tau)**3 - np.sin(theta_s)**3)
                 - d*((tau - theta_s)/2.0 - (np.sin(2.0*tau) - np.sin(2.0*theta_s))/4.0))
        + mu*etan*vn*((tau - theta_s)/2.0 - (np.sin(2.0*tau) - np.sin(2.0*theta_s))/4.0))


# ===========================================================================
#  Case drivers (verification tests 1 - 5)
#
#  Test 1 (normal bounce) is a genuine free-flight dynamics test under gravity.
#  Tests 2 - 5 prescribe a constant velocity / angular velocity and hold the
#  overlap fixed, because the closed-form solutions are only valid for constant
#  kinematics and a constant normal velocity (vn = 0 here). Under free
#  integration the repulsive normal force would simply eject the (unconstrained)
#  sphere and invalidate the formulas. The prescribed-kinematics setup mirrors a
#  held / infinite-inertia body in LAMMPS.
# ===========================================================================
def analytical_solution(t, caseflag):
    """
    Analytical solutions.
    """
    # Manual input, constants, starting pos
    R = 1.0                         # Radius sphere
    density = 2500                  # kg/m3
    g = np.array([0.0, 0.0, -9.81]) # gravitational acceleration, m/s2
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
    X = np.zeros((numSteps, 3))
    V = np.zeros((numSteps, 3))
    F = np.zeros((numSteps, 3))
    T = np.zeros((numSteps, 3))

    # Prescribed V and Omega depend on the case, should be included after the caseflag statement.

    if caseflag == 1:  # Normal linear
        # Free bounce under gravity. Start location
        X[0, :] = np.array([x0, y0, z0])
        # Time stepping
        for i in range(1, numSteps):
            # Approach velocity (positive for approach). The plane is below the
            # sphere at z=0, so moving down (V_z < 0) is approaching: vn = -V_z.
            # NOTE: we no longer clip the velocity here. The stress is clipped
            # to be repulsive instead (see normal_force_linear), which is the
            # correct treatment of a cohesionless contact (Eq. sigmalim).
            vn = -V[i-1, 2]

            if X[i-1, 2] > R:
                # No contact
                F[i, :] = mass*g
            else:
                # Contact
                # Normal bounce
                h = R - X[i-1, 2]  # Cap height [m]
                # Normal component with the repulsive-force (stress) limit
                fz = normal_force_linear(h, vn, R, kn, ketan)
                # Only a z force, no torques
                F[i, :] = np.array([0.0, 0.0, fz]) + mass*g

            # Apply integration of motion
            A = F[i, :] / mass
            V[i, :] = V[i-1, :] + A*dt
            X[i, :] = X[i-1, :] + V[i, :]*dt + 0.5*A*dt*dt

    elif caseflag == 2:  # Tangent linear
        # Tangent velocity and direction follows from set v = (v_x,v_y,0.0), no rotations
        # Prescribed tangent velocity (constant), no vertical velocity
        vx_prescribed = 0.5  # m/s
        vy_prescribed = 0.3  # m/s
        vz_prescribed = 0.0  # m/s (no vertical motion)

        # Held in contact at a constant overlap (vn = 0)
        d = R - 0.05               # constant separation distance, small overlap
        h = R - d
        vn = 0.0                   # constant normal velocity
        vt_mag = np.sqrt(vx_prescribed**2 + vy_prescribed**2)
        nt = np.array([-vx_prescribed, -vy_prescribed, 0.0]) / vt_mag  # tangent direction
        torque_dir = np.array([vy_prescribed, -vx_prescribed, 0.0]) / vt_mag

        # Position drifts linearly under the constant tangent velocity; the
        # overlap (z) is held fixed. Force/torque are translation-invariant, but
        # including the drift makes the position components compare correctly.
        X[:, 0] = x0 + vx_prescribed*t
        X[:, 1] = y0 + vy_prescribed*t
        X[:, 2] = d
        V[:, :] = np.array([vx_prescribed, vy_prescribed, vz_prescribed])

        for i in range(1, numSteps):
            # Normal force (Eq. AFn) with the repulsive-force (stress) limit
            Fn = normal_force_linear(h, vn, R, kn, ketan)
            # Tangent force - partial-slip solution with stick/slip limits
            Ft = tangent_force_linear(h, d, vt_mag, vn, t[i], R, kn, kt, ketan, ketat, mu)
            F[i, :] = np.array([Ft*nt[0], Ft*nt[1], Fn])
            # Torque (partial-slip solution with stick/slip limits)
            Tmag = torque_tangent_linear(h, d, vt_mag, vn, t[i], R, kn, kt, ketan, ketat, mu)
            T[i, :] = Tmag*torque_dir

    elif caseflag == 3:  # Shear angular
        # Here we have omega = (omega_x,omega_y,0) and no linear velocities
        # Prescribed angular velocities (set high enough to reach slip condition)
        omega_x = 5.0  # rad/s
        omega_y = 3.0  # rad/s
        omega_z = 0.0
        omega_mag = np.sqrt(omega_x**2 + omega_y**2)

        # Held in contact at a constant overlap (vn = 0)
        d = R - 0.05
        h = R - d
        tau = np.arccos(d/R)
        vn = 0.0

        X[:, :] = np.array([x0, y0, d])
        V[:, :] = np.array([0.0, 0.0, 0.0])      # no linear velocity

        # Normal force.
        # Under the CORRECT stress limit (sigma_n <- max(0, sigma_n)), a pure
        # spin generates NO net normal force to leading order: the rotational
        # normal velocity is anti-symmetric about the spin axis while the
        # elastic stress is axisymmetric, so the viscous contribution integrates
        # to zero and only a small, higher-order sliver near the receding cap
        # edge is clipped. The earlier velocity-clipped term
        #   Fn_rotation_viscous = ketan*R**3*omega_mag*(tau - sin(2 tau)/2)
        # was an artifact of clipping the velocity (Eq. fnshear) and is dropped.
        # The residual deviation expected versus LAMMPS can be estimated with
        # spurious_normal_force_rotation(omega_mag, d, R, kn, ketan).
        Fn = normal_force_linear(h, vn, R, kn, ketan)

        for i in range(1, numSteps):
            if coulomb_slip_met(omega_mag, d, R, kn, ketat, mu):
                # Maximum shear: the dashpot alone reaches the Coulomb limit
                Ft = max_shear_roll_force(Fn, mu)
                Ft_vec = Ft*np.array([-omega_y, omega_x, 0.0])/omega_mag
                T_vec = torque_max_shear_roll(omega_x, omega_y, d, R, kn, mu)
            else:
                # Below the Coulomb limit: purely viscous solution
                Ft_vec = tangent_force_shear_viscous(omega_x, omega_y, tau, R, ketat)
                T_vec = torque_shear_viscous(omega_x, omega_y, d, R, ketan, ketat)

            F[i, :] = np.array([Ft_vec[0], Ft_vec[1], Fn])
            T[i, :] = T_vec

    elif caseflag == 4:  # Twist, only omega_z is non-zero
        # Solution selector: 'partial', 'stick', or 'slip'
        solution_type = 'partial'  # Hardcoded switch

        # Prescribed twist angular velocity
        omega_z = 10.0  # rad/s

        # Held in contact at a constant overlap (vn = 0)
        d = R - 0.05
        h = R - d
        vn = 0.0

        X[:, :] = np.array([x0, y0, d])
        V[:, :] = np.array([0.0, 0.0, 0.0])      # no linear velocity

        # Normal force (same as case 1) with the repulsive-force (stress) limit
        Fn = normal_force_linear(h, vn, R, kn, ketan)

        for i in range(1, numSteps):
            # No tangent force (symmetry around z-axis)
            F[i, :] = np.array([0.0, 0.0, Fn])
            # Twist torque (partial / stick / slip), always a pure z torque
            T_twist = twist_torque(t[i], omega_z, vn, d, R, kn, kt, ketan, ketat,
                                   mu, mode=solution_type)
            T[i, :] = np.array([0.0, 0.0, T_twist])

    elif caseflag == 5:  # Roll
        # Pure rolling where v_Omega = R*(-omega_y, omega_x, 0)
        omega_x = 2.0  # rad/s
        omega_y = 1.5  # rad/s
        omega_mag = np.sqrt(omega_x**2 + omega_y**2)

        # Linear velocity for pure rolling
        vx_roll = -R*omega_y
        vy_roll = R*omega_x
        vz_roll = 0.0  # No vertical velocity

        # Held in contact at a constant overlap (vn = 0)
        d = R - 0.05
        h = R - d
        tau = np.arccos(d/R)
        vn = 0.0

        # Position drifts linearly under the rolling translation velocity; the
        # overlap (z) is held fixed.
        X[:, 0] = x0 + vx_roll*t
        X[:, 1] = y0 + vy_roll*t
        X[:, 2] = d
        V[:, :] = np.array([vx_roll, vy_roll, vz_roll])

        # Normal force: the rotation-induced normal force vanishes to leading
        # order under the stress limit (see case 3 note); only the bulk normal
        # contribution remains, here just the elastic part since vn = 0.
        Fn = normal_force_linear(h, vn, R, kn, ketan)

        for i in range(1, numSteps):
            # Tangent force for roll (viscous rolling resistance, Eq. ftroll)
            Ft_vec = tangent_force_roll_viscous(omega_x, omega_y, h, R, ketat)
            F[i, :] = np.array([Ft_vec[0], Ft_vec[1], Fn])
            # Roll torque (viscous)
            T[i, :] = torque_roll_viscous(omega_x, omega_y, d, R, ketan, ketat)

    else:
        print("ERROR: INVALID CASEFLAG.")

    # Merge into one array
    results = np.concatenate((X, V, F, T), axis=1)

    return results

# End of file
