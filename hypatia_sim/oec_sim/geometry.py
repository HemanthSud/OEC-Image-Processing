"""
Vectorized orbital mechanics: Walker-delta propagation, ECEF conversion,
elevation angles, and the line-of-sight feasibility test for ISLs.
"""

import numpy as np

from . import config as C


def build_satellites():
    """Return (raan, anom0) arrays of shape (N_SATS,) for the Walker delta."""
    p = np.repeat(np.arange(C.N_PLANES), C.SATS_PER_PLANE)      # plane index
    s = np.tile(np.arange(C.SATS_PER_PLANE), C.N_PLANES)        # in-plane index
    raan = p * (2.0 * np.pi / C.N_PLANES)
    anom0 = (s * (2.0 * np.pi / C.SATS_PER_PLANE)
             + p * C.PHASING_F * (2.0 * np.pi / C.N_SATS))
    return raan, anom0


def sat_ecef(times):
    """ECEF positions for all satellites at all times.

    times: (T,) seconds.  Returns (T, N_SATS, 3) km.
    """
    raan, anom0 = build_satellites()
    t = np.asarray(times, dtype=float)[:, None]                 # (T,1)
    theta = anom0[None, :] + C.N_MM * t                         # (T,N)

    ci, si = np.cos(C.INCL_DEG * C.DEG), np.sin(C.INCL_DEG * C.DEG)
    x_op = C.A_SMA * np.cos(theta)
    y_op = C.A_SMA * np.sin(theta)

    cr, sr = np.cos(raan)[None, :], np.sin(raan)[None, :]
    # ECI = Rz(raan) . Rx(incl) . [x_op, y_op, 0]
    x_eci = cr * x_op - sr * ci * y_op
    y_eci = sr * x_op + cr * ci * y_op
    z_eci = si * y_op

    # ECI -> ECEF (GMST = OMEGA_E * t)
    g = C.OMEGA_E * t
    cg, sg = np.cos(g), np.sin(g)
    x =  cg * x_eci + sg * y_eci
    y = -sg * x_eci + cg * y_eci
    return np.stack([x, y, z_eci], axis=-1)                     # (T,N,3)


def latlon_ecef(lat_deg, lon_deg):
    """ECEF position(s) on the sphere surface, km.  Accepts scalars/arrays."""
    lat = np.asarray(lat_deg) * C.DEG
    lon = np.asarray(lon_deg) * C.DEG
    return C.R_E * np.stack([np.cos(lat) * np.cos(lon),
                             np.cos(lat) * np.sin(lon),
                             np.sin(lat)], axis=-1)


def elevation_deg(sat_pos, gnd_pos):
    """Elevation of satellites above ground-point horizons.

    sat_pos: (T,N,3), gnd_pos: (G,3).  Returns (T,N,G) degrees.
    """
    delta = sat_pos[:, :, None, :] - gnd_pos[None, None, :, :]  # (T,N,G,3)
    dist = np.linalg.norm(delta, axis=-1)                       # (T,N,G)
    n_hat = gnd_pos / np.linalg.norm(gnd_pos, axis=-1, keepdims=True)
    sin_el = np.einsum('tngk,gk->tng', delta, n_hat) / dist
    return np.degrees(np.arcsin(np.clip(sin_el, -1.0, 1.0))), dist


def isl_feasible(pos_a, pos_b):
    """Line-of-sight + range feasibility for satellite pairs.

    pos_a, pos_b: (..., 3) km.  A link is feasible iff
      (1) the segment's closest approach to Earth's centre stays above
          R_E + LOS_GRAZE_KM (no Earth/atmosphere blockage), and
      (2) length <= ISL_MAX_KM.
    Returns (feasible_bool, dist_km) with shape (...,).
    """
    d = pos_b - pos_a
    seg_len = np.linalg.norm(d, axis=-1)
    # closest point on segment to origin: t* = -a.d / |d|^2, clamped to [0,1]
    t_star = -np.einsum('...k,...k->...', pos_a, d) / np.maximum(seg_len**2, 1e-9)
    t_star = np.clip(t_star, 0.0, 1.0)
    closest = pos_a + t_star[..., None] * d
    graze = np.linalg.norm(closest, axis=-1)
    ok = (graze >= C.R_E + C.LOS_GRAZE_KM) & (seg_len <= C.ISL_MAX_KM)
    return ok, seg_len
