"""
Time-varying network topology E_t:
  * +Grid ISLs (each satellite links its in-plane ring neighbours and the
    same-index satellite in the adjacent plane), gated per slot by the
    line-of-sight + max-range feasibility test.
  * GSLs gated by minimum elevation angle.
  * Per-slot shortest-delay routing (Dijkstra from every GBS over the
    feasible graph), giving path and propagation delay for every satellite.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from . import config as C
from . import geometry as G


def build_isl_pairs():
    """+Grid ISL pair list [(a, b), ...] — 2 links per satellite."""
    pairs = []
    for p in range(C.N_PLANES):
        base = p * C.SATS_PER_PLANE
        for s in range(C.SATS_PER_PLANE):
            a = base + s
            pairs.append((a, base + (s + 1) % C.SATS_PER_PLANE))        # intra
            b_plane = (p + 1) % C.N_PLANES
            pairs.append((a, b_plane * C.SATS_PER_PLANE + s))           # inter
    return np.array(sorted(set(tuple(sorted(x)) for x in pairs)))


@dataclass
class Topology:
    times: np.ndarray            # (T,) slot start times, s
    sat_pos: np.ndarray          # (T,N,3) km
    isl_pairs: np.ndarray        # (L,2)
    isl_ok: np.ndarray           # (T,L) bool
    isl_dist: np.ndarray         # (T,L) km
    gsl_ok: np.ndarray           # (T,N,G) bool
    gsl_dist: np.ndarray         # (T,N,G) km
    elev: np.ndarray             # (T,N,G) deg
    aoi_elev: np.ndarray         # (T,N,A) deg
    # routing: per slot, per GBS -> shortest-path tree over feasible graph
    route_dist: np.ndarray = field(default=None)   # (T,G,N) km, inf if cut off
    route_pred: np.ndarray = field(default=None)   # (T,G,N+G) predecessors

    def path(self, t, gs, src_sat):
        """Node path [src_sat, ..., gs_node] at slot t, or None."""
        if not np.isfinite(self.route_dist[t, gs, src_sat]):
            return None
        gs_node = C.N_SATS + gs
        path, node = [src_sat], src_sat
        while node != gs_node:
            node = int(self.route_pred[t, gs, node])
            if node < 0:
                return None
            path.append(node)
        return path

    def delay_ms(self, t, gs, src_sat):
        d = self.route_dist[t, gs, src_sat]
        return float(d) / C.C_KM_S * 1e3 if np.isfinite(d) else None


def build_topology():
    times = np.arange(C.N_SLOTS) * C.SLOT_S
    sat_pos = G.sat_ecef(times)                                  # (T,N,3)

    isl_pairs = build_isl_pairs()
    pa = sat_pos[:, isl_pairs[:, 0], :]
    pb = sat_pos[:, isl_pairs[:, 1], :]
    isl_ok, isl_dist = G.isl_feasible(pa, pb)                    # (T,L)

    gs_pos = G.latlon_ecef([g['lat'] for g in C.GROUND_STATIONS],
                           [g['lon'] for g in C.GROUND_STATIONS])
    elev, gsl_dist = G.elevation_deg(sat_pos, gs_pos)            # (T,N,G)
    gsl_ok = elev >= C.MIN_ELEV_DEG

    aoi_pos = G.latlon_ecef([a['lat'] for a in C.AOIS],
                            [a['lon'] for a in C.AOIS])
    aoi_elev, _ = G.elevation_deg(sat_pos, aoi_pos)              # (T,N,A)

    topo = Topology(times, sat_pos, isl_pairs, isl_ok, isl_dist,
                    gsl_ok, gsl_dist, elev, aoi_elev)
    _compute_routes(topo)
    return topo


def _compute_routes(topo):
    """Dijkstra (distance = delay) from each GBS node at every slot."""
    T = len(topo.times)
    n_nodes = C.N_SATS + C.N_GS
    gs_nodes = np.arange(C.N_SATS, n_nodes)
    topo.route_dist = np.full((T, C.N_GS, C.N_SATS), np.inf)
    topo.route_pred = np.full((T, C.N_GS, n_nodes), -9999, dtype=np.int32)

    for t in range(T):
        ok = topo.isl_ok[t]
        isl_rows = topo.isl_pairs[ok, 0]
        isl_cols = topo.isl_pairs[ok, 1]
        isl_w = topo.isl_dist[t, ok]

        # one graph per GBS (only its own GSL edges), so routes can never
        # transit through a different ground station
        for g in range(C.N_GS):
            s_idx = np.flatnonzero(topo.gsl_ok[t, :, g])
            rows = np.concatenate([isl_rows, s_idx])
            cols = np.concatenate([isl_cols,
                                   np.full(len(s_idx), C.N_SATS + g)])
            w = np.concatenate([isl_w, topo.gsl_dist[t, s_idx, g]])
            graph = csr_matrix((w, (rows, cols)), shape=(n_nodes, n_nodes))
            dist, pred = dijkstra(graph, directed=False,
                                  indices=gs_nodes[g],
                                  return_predecessors=True)
            topo.route_dist[t, g] = dist[:C.N_SATS]
            topo.route_pred[t, g] = pred


def contact_windows(topo):
    """[{gs, sat, start_s, end_s, duration_s}, ...] from gsl_ok."""
    wins = []
    for g in range(C.N_GS):
        for s in range(C.N_SATS):
            vis = topo.gsl_ok[:, s, g]
            if not vis.any():
                continue
            edges = np.flatnonzero(np.diff(np.concatenate([[0], vis, [0]])))
            for i in range(0, len(edges), 2):
                a, b = edges[i], edges[i + 1] - 1
                wins.append({'gs': C.GROUND_STATIONS[g]['name'],
                             'sat': int(s),
                             'start_s': int(topo.times[a]),
                             'end_s': int(topo.times[b]),
                             'duration_s': int(topo.times[b] - topo.times[a])})
    return wins
