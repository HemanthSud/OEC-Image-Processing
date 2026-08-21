"""
MPC-predicted link costs + per-predicted-step Dijkstra (Dr. Liu's directive:
"MPC and Dijkstra are not necessarily mutually exclusive — have MPC predict
link costs, and Dijkstra compute the path at each predicted step").

topology._compute_routes() gives the *geometric* shortest-delay routes used
by the greedy baselines and as the flat MPC's default. PredictiveRouter
below re-runs Dijkstra per predicted horizon step tau, with link cost =
propagation delay + a congestion penalty derived from the MPC's own
predicted load on that link (the load implied by the plan the MILP just
solved). Because the plan depends on the routes and the routes depend on
the plan, MPCScheduler._solve iterates this to a fixed point (method of
successive averages damping + keep-best-iterate, see _solve's outer loop).

Only single-path routing (MPC_ROUTE_NPATHS=1) is implemented: one Dijkstra
tree per (predicted step, GBS) under the current congestion-weighted costs.
Multipath (candidate path sets, MPC_ROUTE_NPATHS>1) is future work.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from . import config as C


class StaticRouter:
    """Wraps the precomputed geometric shortest-delay routes (today's
    behaviour). Used by greedy baselines, and as iteration 0 of the
    fixed point / the fallback when the predictive router fails."""

    def __init__(self, topo):
        self.topo = topo

    def path(self, t, gs, src_sat):
        return self.topo.path(t, gs, src_sat)

    def delay_s(self, t, gs, src_sat):
        ms = self.topo.delay_ms(t, gs, src_sat)
        return (ms / 1e3) if ms is not None else 0.0


class RouteSet:
    """Per-(tau, gbs) shortest-path trees under predicted congestion costs,
    covering absolute slots [t0, t0+H)."""

    def __init__(self, t0, H, n_nodes):
        self.t0 = t0
        self.H = H
        self.dist = np.full((H, C.N_GS, C.N_SATS), np.inf)
        self.pred = np.full((H, C.N_GS, n_nodes), -9999, dtype=np.int32)

    def path(self, tau, g, src_sat):
        gs_node = C.N_SATS + g
        if not np.isfinite(self.dist[tau, g, src_sat]):
            return None
        path, node = [src_sat], src_sat
        for _ in range(300):
            if node == gs_node:
                return path
            node = int(self.pred[tau, g, node])
            if node < 0:
                return None
            path.append(node)
        return None

    def delay_s(self, tau, g, src_sat):
        d = self.dist[tau, g, src_sat]
        return float(d) / C.C_KM_S if np.isfinite(d) else 0.0


def _link_cost(base_dist, load_bits, cap_bits, beta, d_ref, rho_max):
    """w_e = d_e + beta * d_ref * rho_e / (1 - rho_e), rho clipped (never
    delete an edge -- a deleted edge can disconnect the graph and drop the
    task's entire horizon of variables, which is worse than a costly path)."""
    rho = np.clip(load_bits / np.maximum(cap_bits, 1e-9), 0.0, rho_max)
    return base_dist + beta * d_ref * rho / (1.0 - rho)


class PredictiveRouter:
    """Builds a RouteSet for horizon steps [t0, t0+H) from predicted
    per-(tau, edge) load (bits offered that slot), stride-limited so a
    30 s-scale re-plan stays cheap."""

    def __init__(self, topo, isl_index,
                 beta=None, d_ref_km=None, rho_max=None, stride=None):
        self.topo = topo
        self.isl_index = isl_index
        self.beta = C.MPC_ROUTE_BETA if beta is None else beta
        self.d_ref = C.MPC_ROUTE_D_REF_KM if d_ref_km is None else d_ref_km
        self.rho_max = C.MPC_ROUTE_RHO_MAX if rho_max is None else rho_max
        self.stride = C.MPC_ROUTE_STRIDE if stride is None else stride
        self._n_nodes = C.N_SATS + C.N_GS

    def build(self, t0, H, load_bits):
        """load_bits: dict (tau, ('isl',l)|('gsl',u,g)|('gs',g)) -> bits
        predicted to cross that edge during that horizon step."""
        rs = RouteSet(t0, H, self._n_nodes)
        gs_nodes = np.arange(C.N_SATS, self._n_nodes)
        last = {}
        for tau in range(H):
            base_tau = tau - (tau % self.stride)
            if base_tau in last:
                rs.dist[tau] = last[base_tau][0]
                rs.pred[tau] = last[base_tau][1]
                continue
            t = t0 + tau
            ok = self.topo.isl_ok[t]
            rows = self.topo.isl_pairs[ok, 0]
            cols = self.topo.isl_pairs[ok, 1]
            base_isl = self.topo.isl_dist[t, ok]
            isl_idx_ok = np.array(
                [self.isl_index[(min(a, b), max(a, b))]
                 for a, b in zip(rows, cols)])
            isl_load = np.array(
                [load_bits.get((tau, ('isl', l)), 0.0) for l in isl_idx_ok])
            w_isl = _link_cost(base_isl, isl_load,
                                C.ISL_RATE_BPS * C.SLOT_S,
                                self.beta, self.d_ref, self.rho_max)

            dist_t = np.full((C.N_GS, C.N_SATS), np.inf)
            pred_t = np.full((C.N_GS, self._n_nodes), -9999, dtype=np.int32)
            for g in range(C.N_GS):
                s_idx = np.flatnonzero(self.topo.gsl_ok[t, :, g])
                base_gsl = self.topo.gsl_dist[t, s_idx, g]
                gsl_load = np.array(
                    [load_bits.get((tau, ('gsl', int(s), g)), 0.0)
                     for s in s_idx])
                agg_load = load_bits.get((tau, ('gs', g)), 0.0)
                # per-sat GSL edge cost + a share of the aggregate GBS
                # congestion attributed evenly across its active senders
                gs_load_each = (agg_load / max(len(s_idx), 1))
                w_gsl = _link_cost(base_gsl, gsl_load + gs_load_each,
                                    min(C.GSL_RATE_BPS, C.GS_RATE_BPS) * C.SLOT_S,
                                    self.beta, self.d_ref, self.rho_max)
                g_rows = np.concatenate([rows, s_idx])
                g_cols = np.concatenate([cols, np.full(len(s_idx), C.N_SATS + g)])
                g_w = np.concatenate([w_isl, w_gsl])
                graph = csr_matrix((g_w, (g_rows, g_cols)),
                                    shape=(self._n_nodes, self._n_nodes))
                dist, pred = dijkstra(graph, directed=False,
                                      indices=C.N_SATS + g,
                                      return_predecessors=True)
                dist_t[g] = dist[:C.N_SATS]
                pred_t[g] = pred
            rs.dist[tau] = dist_t
            rs.pred[tau] = pred_t
            last[base_tau] = (dist_t, pred_t)
        return rs
