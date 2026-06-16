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

    Damping / repulsion model: the local stress sigma_n = kn*delta + etan*vn is
    formed with the SIGNED normal velocity and then clipped to be repulsive,
        sigma_n <- max(0, sigma_n)        (Eq. sigmalim) .
    This matches the LS-DEM implementation, which forms the per-node force with
    the signed velocity and caps the total at zero,
        fn = MAX(kn*u + etan*v_rel_n, 0)  (pair_ls_dem.cpp:521-535) .
    On separation (vn < 0) the dashpot is therefore active and reduces the
    force; the outer ring of the cap (smallest elastic stress) goes tensile
    first and is clipped, while the interior still carries a reduced force. The
    repulsive region extends to an effective cap height h_eff where sigma_n = 0,
    and integrating the clipped stress over the cap gives pi*R*kn*h_eff^2.

    NOTE: clip the FORCE (max(Fn,0)), NOT the velocity (max(vn,0)). Velocity-
    clipping switches the dashpot off entirely on the way out and overshoots the
    restitution; force-clipping keeps the (repulsive part of the) dashpot, which
    is what the code does.

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
    (Eq. T_t), with stick and slip limits. The direction is (-vy, vx, 0)
    (built by the caller; lever arm r = (0, 0, -R) acting on the friction force
    (-vx, -vy) gives T = r x F ~ (-vy, vx)); here we return the scalar magnitude
    that multiplies that (unit) direction.

    LEVER-ARM CORRECTION (2026-06-16): the in-plane torque uses the geometric
    contact-point lever a(theta) = (R*cos(theta) + d)/2 (the z-distance from the
    COM to the contact point, which sits at the midpoint between the surfaces --
    exactly what the code uses, contact_point = node - 0.5*u*normal). The
    manuscript instead used the local overlap delta = R*cos(theta) - d as the
    lever (its bracket [R^2/3(1-c^3) - Rd(1-c^2) + d^2(1-c)] = integral delta^2),
    which is O(h^3) and ~20-60x too small. With a(theta) the torque of an
    in-plane traction tau_t is
        |T| = 2 pi R^2 integral_0^tau a(theta) tau_t(theta) sin(theta) dtheta ,
    which is O(h^2) ~ mu*Fn*<a> and matches the sim.
    """
    tau = np.arccos(np.clip(d/R, -1.0, 1.0))
    c = np.cos(tau)
    drive = (kt*t + etat)*vt                       # uniform tangential stress
    lo = drive - mu*etan*vn
    if lo <= 0.0:
        # Stick limit (theta_s -> tau): elastic stress 'drive' over whole cap.
        return np.pi*R**2*drive*(R*np.sin(tau)**2/2.0 + d*(1.0 - c))
    if lo >= mu*kn*h:
        # Slip limit (theta_s -> 0): traction mu*kn*(R cos theta - d) everywhere.
        return np.pi*R**2*mu*kn*(R**2*(1.0 - c**3)/3.0 - d**2*(1.0 - c))
    # Partial slip: stick (elastic 'drive') for 0<theta<theta_s, slip
    # (mu*kn*delta) for theta_s<theta<tau; xi = cos(theta_s).
    xi = (drive - mu*etan*vn + mu*kn*d)/(mu*kn*R)
    return 2.0*np.pi*R**2*(
        drive*(R*(1.0 - xi**2)/4.0 + d*(1.0 - xi)/2.0)
        + (mu*kn/2.0)*(R**2*(xi**3 - c**3)/3.0 - d**2*(xi - c)))


# --------------------------- Shear / roll: angular velocity ----------------
def coulomb_slip_met(Omega, d, R, kn, etat, mu):
    """Maximum shear/roll condition (Eq. CoulombLim): the dashpot alone reaches
    the Coulomb limit over the whole cap."""
    return Omega >= (mu*kn/etat)*(1.0 - d/R)


def tangent_force_shear_viscous(omega_x, omega_y, tau, R, etat):
    """
    Purely viscous tangent force for pure shear (v_Omega = 0).

    The contact-point surface velocity from the spin is omega x r with the
    lever arm r = (0, 0, -R) (contact BELOW the centre), giving R*(-omega_y,
    omega_x, 0). Friction OPPOSES it, so the force direction is
    (omega_y, -omega_x, 0). (The manuscript wrote (-omega_y, omega_x); that is
    the surface-velocity direction, i.e. the wrong sign for the force.)
    """
    return np.pi*etat*R**3*np.sin(tau)**2 * np.array([omega_y, -omega_x, 0.0])


def tangent_force_roll_viscous(omega_x, omega_y, h, R, etat):
    """
    Viscous rolling resistance (Eq. ftroll) for TRUE rolling (no contact slip).

    With the no-slip translation v_cm = R(omega_y, -omega_x) the contact point is
    instantaneously stationary, but cap elements at angle theta still slide at
    |v_rel_t| = R*Omega*(1 - cos theta); the viscous traction opposes this. The
    net in-plane direction is (-omega_y, omega_x) -- i.e. it OPPOSES the rolling
    velocity v_cm = R(omega_y, -omega_x), which is the sign the case-5 sim shows.
    (The old (omega_y, -omega_x) was for the earlier slip setup and is wrong for
    true rolling.)

    Magnitude pi*etat*R*h^2*Omega = etat*R*Omega*2 pi R^2 [(1-cos tau) -
    sin^2 tau/2] is the ideal viscous integral. The sim is larger (~5x at this
    overlap) because of elastic shear accumulated as nodes traverse the contact
    and the non-objective v_rel_t in the pair style; neither is captured here.
    """
    return np.pi*etat*R*h*h * np.array([-omega_y, omega_x, 0.0])


def tangent_force_roll_elastic(omega_x, omega_y, tau, R, kt):
    """
    ELASTIC rolling resistance -- the component the manuscript omits (it derives
    only the viscous roll force, Eq. ftroll, and explicitly forgoes the elastic
    part). Derived and validated 2026-06-16 against the case-5 sim (etat = 0).

    In steady true rolling a surface node enters the contact patch
    (radius a = R sin tau) at the leading edge with zero shear and convects
    through it at the rolling speed v = R*Omega; the local slip
    R*Omega*(1 - cos theta) integrates along the path to an elastic shear
    DISPLACEMENT (independent of Omega, hence purely elastic)
        u_s(x') = (a^3 - x'^3) / (6 R^2) ,
    with x' the coordinate along the rolling direction. Integrating the elastic
    shear stress tau_s = kt*u_s over the circular patch (the odd x'^3 term
    integrates to zero) gives a net resisting force
        |F_t,el^roll| = pi*kt*a^5/(6 R^2) = pi*kt*R^3*sin^5(tau)/6 ,
    directed opposite the rolling velocity, i.e. along (-omega_y, omega_x).

    Validation (N=400, kt=0.5e8): derived 77.7e3 vs sim 84.7e3 (~1.09, the same
    mesh-resolution factor as the forces). It dominates the viscous roll force
    (~19.6e3) by ~4x, which is why the viscous-only model underpredicts the sim.
    """
    Omega = np.hypot(omega_x, omega_y)
    mag = np.pi*kt*R**3*np.sin(tau)**5/6.0
    return mag*np.array([-omega_y, omega_x, 0.0])/Omega


def torque_roll_elastic(omega_x, omega_y, d, R, kt):
    """
    Torque of the elastic rolling resistance: the contact-point lever
    a = (R cos theta + d)/2 acting on F_t,el^roll. To leading order in the
    overlap a -> (R + d)/2, giving magnitude ((R+d)/2)*|F_t,el^roll| along
    (omega_x, omega_y) (= lever x force, same sense as the viscous roll torque
    tangential term).
    """
    Omega = np.hypot(omega_x, omega_y)
    tau = np.arccos(np.clip(d/R, -1.0, 1.0))
    mag = (R + d)/2.0 * np.pi*kt*R**3*np.sin(tau)**5/6.0
    return mag*np.array([omega_x, omega_y, 0.0])/Omega


def max_shear_roll_force(Fn, mu):
    """Maximum shear/roll tangent force magnitude (= mu*Fn = Ft_slip)."""
    return mu*Fn


def torque_max_shear_roll(omega_x, omega_y, d, R, kn, mu):
    """
    Maximum torque due to SHEAR (differential surface velocity -> sliding) at
    the Coulomb limit. (True rolling has no contact slip, so it is not at the
    Coulomb limit; the rolling-resistance torque is handled by the viscous roll
    functions, despite this function's historical name.)

    Direction from the lever arm r = (0, 0, -R): the slip traction is along the
    friction force (omega_y, -omega_x), so T = r x F is along (-omega_x,
    -omega_y).

    LEVER-ARM CORRECTION (2026-06-16): lever arm is the contact-point distance
    a(theta) = (R*cos(theta) + d)/2, not the local overlap delta. With traction
    mu*kn*(R cos theta - d) opposing the uniform in-plane slip direction,
        |T| = pi R^2 mu kn [R^2(1-cos^3 tau)/3 - d^2(1-cos tau)]
    (the 1/Omega cancels against |(-omega_x,-omega_y)| = Omega). This is O(h^2)
    ~ mu*Fn*<a> and matches the sim; the old overlap-lever form was O(h^3) and
    ~20-60x too small. NB the direction is (-omega_x,-omega_y); the manuscript's
    (-omega_y,-omega_x) (line ~1073) has the components swapped.
    """
    Omega = np.hypot(omega_x, omega_y)
    tau = np.arccos(np.clip(d/R, -1.0, 1.0))
    c = np.cos(tau)
    mag = np.pi*mu*kn*R**2*(R**2*(1.0 - c**3)/3.0 - d**2*(1.0 - c))
    return mag*np.array([-omega_x, -omega_y, 0.0])/Omega


def normal_viscous_torque(omega_x, omega_y, d, R, etan):
    """
    Normal-viscous contribution to the torque from rotation about a horizontal
    axis (the contact-normal / z component of the viscous stress).

    This term is IDENTICAL for pure shear and for pure roll: in rolling, the
    translation v_Omega = R(-omega_y, omega_x, 0) is purely in-plane and adds
    nothing to the z-velocity of the cap, so the mechanism that generates this
    torque is the same in both cases. Derived and verified (symbolically and by
    direct 2D integration) as
        T = pi * etan * R^4 * (2/3 - cos tau + cos^3 tau / 3) * (-omega_x, -omega_y, 0).

    Direction: this is a DAMPING torque and must oppose the spin. The normal
    dashpot resists the differential z-velocity v_z = omega_x*y across the cap,
    producing T_x = integral(y*F_z) ~ -omega_x (and likewise T_y ~ -omega_y).
    So the direction is (-omega_x, -omega_y, 0). (The manuscript's
    (omega_x, omega_y) had the wrong sign -- it would be an anti-damping torque;
    the case-5 roll dump confirms the (-omega_x, -omega_y) sense.)

    NOTE (manuscript correction): the manuscript wrote this term as
    pi*etan*R^5/2*(...) for shear and pi/2*etan*R^4*(...) for roll. Both are
    wrong: the power is R^4 (not R^5; the R^5 was invisible in the tests because
    R = 1), and the coefficient is pi (not pi/2). The R^5-vs-R^4 mismatch
    between the two manuscript expressions was the tell.

    This is the UNCAPPED form. The non-tensile stress limit (Eq. sigmalim)
    reduces it modestly (to roughly 0.87 of this value for the shear test and
    0.94 for the roll test) by clipping a thin sliver near the receding cap
    edge. Unlike the rotational normal FORCE (which nearly vanishes because its
    integrand is odd in the azimuth), this torque integrand is even in the
    azimuth and therefore survives the cap largely intact.
    """
    tau = np.arccos(np.clip(d/R, -1.0, 1.0))
    g = 2.0/3.0 - np.cos(tau) + np.cos(tau)**3/3.0
    return np.pi*etan*R**4*g*np.array([-omega_x, -omega_y, 0.0])


def torque_shear_viscous(omega_x, omega_y, d, R, etan, etat):
    """Viscous torque for pure shear (T_eta)."""
    tau = np.arccos(np.clip(d/R, -1.0, 1.0))
    # Tangential (eta_t) contribution. Direction (-omega_x, -omega_y) from the
    # contact-point lever acting on the shear friction force (omega_y, -omega_x).
    # The d-term sign is + (contact-point lever a=(R cos theta + d)/2); the
    # manuscript's - sign is the same overlap-lever error as the slip torques.
    tangential = (2.0*np.pi*etat*R**3*(R*(1.0 - np.cos(tau)**3)/6.0
                                       + d*np.sin(tau)**2/4.0)
                  )*np.array([-omega_x, -omega_y, 0.0])
    # Normal-viscous (eta_n) contribution, corrected and shared with roll.
    return tangential + normal_viscous_torque(omega_x, omega_y, d, R, etan)


def torque_roll_viscous(omega_x, omega_y, d, R, etan, etat):
    """Viscous torque for pure roll (T_eta^roll)."""
    tau = np.arccos(np.clip(d/R, -1.0, 1.0))
    # Tangential (eta_t) rolling-resistance contribution, transcribed from the
    # manuscript. Its leading minus sign combines with the bracket (positive for
    # a small overlap) to give the (-omega_x, -omega_y) direction -- consistent
    # with the lever arm r = (0, 0, -R) acting on the roll force (omega_y,
    # -omega_x). This term is left as-is: it already points the right way (the
    # 2026-06-11 audit's proposed extra negation here would double-flip it).
    tangential = (2.0*np.pi*etat*R**3*(R*(1.0 - 3.0*np.cos(tau)**2 + 2.0*np.cos(tau)**3)/12.0
                                        + d*(1.0 - np.cos(tau))**2/4.0)
                   )*np.array([omega_x, omega_y, 0.0])
    # Normal-viscous (eta_n) contribution is the SAME as for shear (see helper),
    # now corrected to point in (-omega_x, -omega_y) as a damping torque. It is
    # added with a + sign in both callers; the helper carries the sign.
    return tangential + normal_viscous_torque(omega_x, omega_y, d, R, etan)


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
        torque_dir = np.array([-vy_prescribed, vx_prescribed, 0.0]) / vt_mag

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
                # Friction opposes the contact-point surface velocity
                # R*(-omega_y, omega_x); direction (omega_y, -omega_x).
                Ft_vec = Ft*np.array([omega_y, -omega_x, 0.0])/omega_mag
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
        # TRUE rolling (no contact slip): the no-slip condition
        # v_cm + omega x r_contact = 0, with r_contact = (0, 0, -R), gives
        # v_cm = R*(omega_y, -omega_x, 0). The sim confirms zero contact slip.
        omega_x = 2.0  # rad/s
        omega_y = 1.5  # rad/s
        omega_mag = np.sqrt(omega_x**2 + omega_y**2)

        # Linear velocity for pure (no-slip) rolling
        vx_roll = R*omega_y
        vy_roll = -R*omega_x
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
            # Tangent force for roll = viscous rolling resistance (Eq. ftroll) +
            # the elastic rolling resistance (derived here, dominant ~4x). The
            # manuscript gives only the viscous part; the elastic part is needed
            # to match the sim. Set the elastic term to 0 to recover the
            # manuscript's viscous-only solution.
            Ft_vec = (tangent_force_roll_viscous(omega_x, omega_y, h, R, ketat)
                      + tangent_force_roll_elastic(omega_x, omega_y, tau, R, kt))
            F[i, :] = np.array([Ft_vec[0], Ft_vec[1], Fn])
            # Roll torque = viscous (eta_t tangential + eta_n normal) + elastic.
            T[i, :] = (torque_roll_viscous(omega_x, omega_y, d, R, ketan, ketat)
                       + torque_roll_elastic(omega_x, omega_y, d, R, kt))

    else:
        print("ERROR: INVALID CASEFLAG.")

    # Merge into one array
    results = np.concatenate((X, V, F, T), axis=1)

    return results

# End of file
