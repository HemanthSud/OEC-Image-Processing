"""Unified utility — the single source of truth for "how good is a schedule?".

Historically this simulator defined utility in five places that disagreed:
the realized score (schedulers._record_delivery) had only weight x freshness
x depth-quality; the flat MPC objective added a backlog bonus and a tardiness
penalty; the hierarchical upper level had neither; its lower level swapped the
backlog bonus for a Lyapunov backpressure term; and the offline bound had a
third combination again. Timeliness, coverage and depth mix lived in side
tables, so no single number said whether a scheduler was actually good.

Every one of those call sites now goes through this module.

    U_k = wbar_k [ omega_Q s_q Ghat_k  -  omega_T T_k  -  omega_E C_k ]
    U   = sum_k U_k  +  omega_F |K| min_k Uhat_k

  s_q      downstream segmentation quality at depth q (mIoU-derived), or the
           legacy reconstruction proxy 1 - LPIPS_q
  Ghat_k   coverage gain in [0,1]: a CONCAVE function of delivered fraction,
           weighted by the freshness at which each image arrived
  T_k      tardiness in [0,1], clipped at one deadline-unit of lateness
  C_k      resource cost in [0,1]: payload-proportional + per-image fixed
  Uhat_k   U_k normalized so the fairness floor is scale- and weight-free

Legacy identity: with omega_Q=1, omega_T=omega_E=omega_F=0, a single coverage
segment of unit width and unit slope, no weight normalization and
s_q = 1 - LPIPS_q, this collapses *literally* to the old
    sum_k sum_t w_k phi_k(t_arr) u_q dr/N_k.
Legacy mode is therefore a parameter setting, not a code branch, which is why
the committed numbers (mpc 64.95, greedy-fixed-8 64.15, ...) stay exactly
reproducible via --utility legacy.

See FORMULATION.md for the MILP linearizations (concave coverage without
binaries; the maximin fairness surrogate) and why the offline bound stays
valid under every added term.
"""

import json
import os

import numpy as np

from . import config as C

_QUALITY_CACHE = None


# ── data side: pure functions of config, no decision variables ───────────────

def load_quality_table(path=None, force=False):
    """Downstream quality table, preferring the JSON written by the FLAIR
    segmentation sweep (rq-vae/downstream/harvest_metrics.py) and falling back
    to the PROVISIONAL dict in config so run_all works with no server access.
    The returned dict always carries a 'source' string; print it wherever the
    numbers are reported so provisional runs are never mistaken for measured.
    """
    global _QUALITY_CACHE
    if _QUALITY_CACHE is not None and not force:
        return _QUALITY_CACHE
    path = path or C.QUALITY_TABLE_PATH
    table = None
    if os.path.exists(path):
        with open(path) as fh:
            raw = json.load(fh)
        anchor = C.UTIL_QUALITY_ANCHOR
        miou = {int(k): float(v) for k, v in raw.get('miou', {}).items()}
        ref, floor = raw.get('miou_ref'), raw.get('miou_floor')
        if anchor == 'floor' and miou and ref is not None and floor is not None:
            # The decision-theoretic zero for a scheduler is not "mIoU = 0" but
            # "what you get by NOT delivering the image" -- i.e. the segmenter
            # running on NIR + Elevation with the RGB bands blanked. Anchoring
            # there is what turns a compressed ~12%-wide quality spread into a
            # spread wide enough for the depth decision to matter.
            span = float(ref) - float(floor)
            s = {q: (v - float(floor)) / span for q, v in miou.items()}
        elif anchor == 'ratio' and miou and ref:
            s = {q: v / float(ref) for q, v in miou.items()}
        else:
            s = {int(k): float(v) for k, v in raw.get('s', {}).items()}
        table = dict(raw)
        table['s'] = s
    if table is None:
        table = dict(C.QUALITY_TABLE_FALLBACK)
        table['s'] = {int(k): float(v) for k, v in table['s'].items()}
    _QUALITY_CACHE = table
    return table


def quality_source_label():
    """One-line provenance for summary.txt / plots."""
    if C.UTIL_QUALITY_SOURCE == 'lpips':
        return 'u_q = 1 - LPIPS_q (truncation_eval.txt, 7050 FLAIR val images)'
    t = load_quality_table()
    return f"s_q = downstream mIoU [{t.get('anchor', '?')}-anchored] :: {t.get('source', '?')}"


def quality_table():
    """{depth: s_q} for the configured quality source."""
    if C.UTIL_QUALITY_SOURCE == 'lpips':
        return dict(C.UTILITY)
    s = load_quality_table()['s']
    missing = [q for q in C.DEPTHS if q not in s]
    if missing:
        raise KeyError(f'quality table has no entry for depths {missing}; '
                       f'source={load_quality_table().get("source")}')
    return {q: s[q] for q in C.DEPTHS}


def quality(q):
    return quality_table()[q]


def quality_max():
    return max(quality_table().values())


def weight(task):
    """w_k, normalized to (0,1] in unified mode so the fairness floor and the
    per-term weights are comparable across tasks."""
    if C.UTIL_WEIGHT_NORM:
        return task.weight / float(max(C.TASK_WEIGHTS))
    return float(task.weight)


# ── coverage: concave, and linear in the MILP ────────────────────────────────

def segments():
    """[(width, slope)] of the piecewise-linear concave coverage curve g(rho),
    g(0)=0, g(1)=1, slopes strictly decreasing."""
    return list(C.UTIL_COVERAGE_BREAKS)


def n_segments():
    return len(C.UTIL_COVERAGE_BREAKS)


def residual_widths(rho_now):
    """Widths of each coverage segment still unclaimed at coverage rho_now.

    Essential for the rolling horizon: coverage already delivered before this
    re-plan has consumed the early, high-slope segments. Without this the MPC
    re-earns segment-1 credit at every single re-plan and systematically
    over-values whichever task it looks at next.
    """
    out, cum = [], 0.0
    for width, _slope in segments():
        lo, hi = cum, cum + width
        out.append(max(0.0, min(hi, 1.0) - max(rho_now, lo)))
        cum = hi
    return out


class CoverageAccumulator:
    """Realized-side counterpart of the MILP's lambda block: fills coverage
    segments chronologically, so image number one is credited at the steepest
    slope and the last image at the shallowest.

    The MILP is free to assign deliveries to segments in any order, but its
    optimum coincides with this chronological fill: the sub-problem is a
    transportation problem with cost m_j * phi_tau where m_j decreases in j and
    phi_tau is non-increasing in tau, so by the rearrangement inequality the
    north-west-corner assignment is optimal. That is why no ordering
    constraints are needed in the MILP (see FORMULATION.md).
    """

    __slots__ = ('rho', 'gain')

    def __init__(self, rho=0.0):
        self.rho = float(rho)
        self.gain = 0.0

    def add(self, images_frac, phi):
        """Credit a delivery of `images_frac` (of N_k) arriving at freshness
        `phi`; returns the increment to Ghat_k."""
        remaining, cum, gained = float(images_frac), 0.0, 0.0
        for width, slope in segments():
            lo, hi = cum, cum + width
            cum = hi
            avail = min(hi, 1.0) - max(self.rho, lo)
            if avail <= 0.0:
                continue
            take = min(remaining, avail)
            if take <= 0.0:
                break
            gained += slope * phi * take
            self.rho += take
            remaining -= take
            if remaining <= 1e-15:
                break
        self.gain += gained
        return gained


# ── the remaining per-image coefficients ─────────────────────────────────────

def tardiness_coeff(task, t_arrive):
    """T_k's per-unit-coverage coefficient, in [0,1]. The clip is applied to a
    *coefficient* (arrival times are data, not decision variables), so it costs
    nothing in MILP linearity."""
    late_s = max(0.0, t_arrive - task.deadline_s)
    return min(1.0, late_s / C.TARDINESS_REF_S)


def cost_coeff(q):
    """C_k's per-unit-coverage coefficient, in [0,1]: a payload-proportional
    part plus a per-image fixed part (the depth-independent encode)."""
    b_max = C.PAYLOAD_B[max(C.DEPTHS)]
    return (C.UTIL_W_COST_TX * C.PAYLOAD_B[q] / b_max
            + C.UTIL_W_COST_ENC * 1.0)


def energy_joules(images, q):
    """(encode J, downlink J) for `images` images at depth q. Accounting only:
    absolute Joules are reported, never optimized -- see the note in config."""
    enc = images * (C.ENC_S_PER_IMG + C.ENC_S_PER_STAGE * q) * C.ENC_POWER_W
    tx = images * C.PAYLOAD_B[q] * 8 * C.E_PER_BIT_J
    return enc, tx


def objective_tardiness_coeff(task, t_arrive):
    """The legacy objective-only tardiness regularizer (MPC_LAMBDA_LATE). It
    was never part of the reported score, so it stays separate from T_k;
    apply_utility_mode('unified') zeroes MPC_LAMBDA_LATE because omega_T then
    covers lateness in both the objective and the score."""
    if C.MPC_LAMBDA_LATE == 0.0:
        return 0.0
    late_s = max(0.0, t_arrive - task.deadline_s)
    return (C.MPC_LAMBDA_LATE * task.weight
            * late_s / C.TARDINESS_REF_S / task.n_images)


# ── realized side ────────────────────────────────────────────────────────────

def _acc(task):
    acc = getattr(task, '_cov_acc', None)
    if acc is None:
        acc = CoverageAccumulator()
        task._cov_acc = acc
    return acc


def accumulate(task, images, t_arrive):
    """Credit a delivery and return the increment to U_k. This is the whole of
    the realized score; schedulers._record_delivery calls nothing else."""
    frac = images / task.n_images
    phi = task.freshness(t_arrive)
    gain = _acc(task).add(frac, phi)
    w = weight(task)
    d = w * C.UTIL_W_QUALITY * quality(task.depth) * gain
    if C.UTIL_W_TARDY:
        d -= w * C.UTIL_W_TARDY * tardiness_coeff(task, t_arrive) * frac
    if C.UTIL_W_COST:
        d -= w * C.UTIL_W_COST * cost_coeff(task.depth) * frac
    task._u_quality = getattr(task, '_u_quality', 0.0) + \
        w * C.UTIL_W_QUALITY * quality(task.depth) * gain
    task._u_tardy = getattr(task, '_u_tardy', 0.0) + \
        (w * C.UTIL_W_TARDY * tardiness_coeff(task, t_arrive) * frac
         if C.UTIL_W_TARDY else 0.0)
    task._u_cost = getattr(task, '_u_cost', 0.0) + \
        (w * C.UTIL_W_COST * cost_coeff(task.depth) * frac
         if C.UTIL_W_COST else 0.0)
    return d


def reset(task):
    """Clear per-task utility state (used when a task list is reused)."""
    for attr in ('_cov_acc', '_u_quality', '_u_tardy', '_u_cost'):
        if hasattr(task, attr):
            delattr(task, attr)


def normalized(task):
    """Uhat_k -- U_k rescaled so the fairness floor is scale-free AND
    weight-relative: each task is measured against a comparable fraction of
    *its own* achievable value, so the floor cannot be gamed by starving
    low-weight AOIs, nor does it perversely starve high-weight ones."""
    denom = C.UTIL_W_QUALITY * quality_max()
    if denom <= 0.0:
        return 0.0
    return task.delivered_utility / (weight(task) * denom) if weight(task) else 0.0


def components(task):
    """Per-task utility breakdown for task_outcomes_*.csv."""
    acc = getattr(task, '_cov_acc', None)
    enc_j, tx_j = energy_joules(task.delivered, task.depth or max(C.DEPTHS))
    return {
        'u_quality': getattr(task, '_u_quality', 0.0),
        'u_tardiness': getattr(task, '_u_tardy', 0.0),
        'u_cost': getattr(task, '_u_cost', 0.0),
        'u_coverage_gain': acc.gain if acc else 0.0,
        'u_norm': normalized(task),
        'enc_energy_j': enc_j,
        'tx_energy_j': tx_j,
    }


def run_utility(tasks):
    """(total U, per-term totals, Jain index, u_min).

    The fairness bonus makes the total differ from sum_k U_k, which is exactly
    the point: a scheduler cannot buy total utility by starving a task.
    Jain is reported as a *diagnostic only* -- it is a ratio of quadratics and
    is not MILP-representable, so optimizing it would leave the offline bound
    with nothing valid to compare against. The maximin floor u_min is the
    surrogate that is actually optimized.
    """
    per = [t.delivered_utility for t in tasks]
    total = sum(per)
    u_norm = [normalized(t) for t in tasks] or [0.0]
    u_min = min(u_norm)
    if C.UTIL_W_FAIR:
        total += C.UTIL_W_FAIR * len(tasks) * u_min
    ssq = sum(u * u for u in u_norm)
    jain = (sum(u_norm) ** 2) / (len(u_norm) * ssq) if ssq > 0 else 0.0
    terms = {
        'quality': sum(getattr(t, '_u_quality', 0.0) for t in tasks),
        'tardiness': sum(getattr(t, '_u_tardy', 0.0) for t in tasks),
        'cost': sum(getattr(t, '_u_cost', 0.0) for t in tasks),
        'fairness': C.UTIL_W_FAIR * len(tasks) * u_min,
    }
    enc = sum(energy_joules(t.delivered, t.depth or max(C.DEPTHS))[0] for t in tasks)
    tx = sum(energy_joules(t.delivered, t.depth or max(C.DEPTHS))[1] for t in tasks)
    terms['enc_energy_j'], terms['tx_energy_j'] = enc, tx
    return total, terms, jain, u_min


# ── MILP side: coefficients shared by schedulers.py, hier.py, oracle.py ──────

def y_coeff(task, q, t_arrive, w=None):
    """Coefficient on y[k,q,tau] for every term that is linear in images.

    When there is a single coverage segment (legacy) the quality term is folded
    in here too, so the lambda block can be skipped entirely and legacy mode
    pays nothing for the concave machinery.
    """
    w = weight(task) if w is None else w
    c = 0.0
    if n_segments() == 1:
        slope = segments()[0][1]
        c += (C.UTIL_W_QUALITY * w * quality(q)
              * slope * task.freshness(t_arrive))
    if C.UTIL_W_TARDY:
        c -= C.UTIL_W_TARDY * w * tardiness_coeff(task, t_arrive)
    if C.UTIL_W_COST:
        c -= C.UTIL_W_COST * w * cost_coeff(q)
    return c / task.n_images


def seg_coeff(task, q, t_arrive, j, w=None):
    """Coefficient on lambda[k,j,tau] (already a *fraction* of N_k, so no
    1/N_k here). Only used when n_segments() > 1."""
    w = weight(task) if w is None else w
    slope = segments()[j][1]
    return C.UTIL_W_QUALITY * w * quality(q) * slope * task.freshness(t_arrive)


def fair_row_coeffs(task, realized=None):
    """(const, scale) for the maximin row  u_min <= const + scale * (in-horizon
    utility of task k).  Uhat_k must be CUMULATIVE -- realized plus planned --
    or a task admitted late always looks starved and the term becomes noise.

    `realized` overrides the already-delivered utility that forms the
    constant. The offline bound MUST pass 0: it is handed an
    already-simulated task list but re-plans every task from scratch, so
    crediting delivered_utility as well would let u_min reach ~2 instead of
    1 and push the bound above the analytic ceiling -- i.e. make it not a
    bound at all.
    """
    denom = C.UTIL_W_QUALITY * quality_max() * weight(task)
    if denom <= 0.0:
        return 0.0, 0.0
    r = task.delivered_utility if realized is None else realized
    return r / denom, 1.0 / denom


def stride_grouper():
    """group_of for SegmentBlock that folds UTIL_SEG_STRIDE consecutive
    horizon steps into one coverage-segment column."""
    stride = max(1, int(C.UTIL_SEG_STRIDE))
    if stride == 1:
        return None
    return lambda key: (key[0], key[1], key[2] // stride)


class SegmentBlock:
    """The concave-coverage lambda block, shared by all three MILP builders.

    Given a y-index map whose keys are (kid, q, s) -- s being a horizon step
    (schedulers, hier lower) or a window index (hier upper, oracle) -- this
    emits the columns and rows that turn the linear coverage term into the
    concave g(rho):

        sum_j lambda[g,j] == sum_{key in g} y[key] / N_k        (linking)
        sum_{keys}  lambda[g,j] <= residual_width_j(rho_now)    (segment cap)

    No binaries and no SOS2: the slopes are strictly decreasing and the
    objective is maximized, so the LP relaxation fills segment 1 before
    segment 2 and the concave envelope is exact at every vertex.

    `group_of` lets the lambda columns be indexed more coarsely than y (the
    offline bound uses ORACLE_SEG_AGG for this). Coarser grouping only gives
    the solver MORE freedom, so a bound built this way stays a valid bound.

    When there is a single coverage segment the block is inert -- no columns,
    no rows -- and utility.y_coeff folds the quality term back onto y, so
    legacy mode pays nothing at all for this machinery.
    """

    __slots__ = ('n_seg', 'li', 'group_of', '_members')

    def __init__(self, yi, base, group_of=None):
        self.n_seg = n_segments()
        self.group_of = group_of or (lambda key: key)
        self.li = {}
        self._members = {}
        if self.n_seg <= 1:
            return
        for key in yi:
            self._members.setdefault(self.group_of(key), []).append(key)
        for g in self._members:
            for j in range(self.n_seg):
                self.li[(g, j)] = base + len(self.li)

    @property
    def active(self):
        return self.n_seg > 1

    def __len__(self):
        return len(self.li)

    def set_objective(self, c, task_by_id, t_arrive_of, w_of=None, val=None):
        """t_arrive_of(group_key) -> arrival time used for the freshness.
        `val`, if given, receives the same coefficients un-negated: it is the
        pure reported-score part, which the fairness rows need separated from
        the objective-only regularizers."""
        for (g, j), i in self.li.items():
            kid, q = g[0], g[1]
            k = task_by_id[kid]
            w = w_of(k) if w_of is not None else None
            coef = seg_coeff(k, q, t_arrive_of(g), j, w=w)
            c[i] = -coef
            if val is not None:
                val[i] = coef

    def columns_of(self, kid):
        """Column indices belonging to one task (for the fairness rows)."""
        return [i for (g, _j), i in self.li.items() if g[0] == kid]

    def add_rows(self, add, ub_var, yi, task_by_id, rho_of=None):
        if not self.active:
            return
        for g, keys in self._members.items():
            k = task_by_id[g[0]]
            ent = [(self.li[(g, j)], 1.0) for j in range(self.n_seg)]
            ent += [(yi[key], -1.0 / k.n_images) for key in keys]
            add(ent, 0.0, 0.0)
        by_task = {}
        for g in self._members:
            by_task.setdefault(g[0], []).append(g)
        for kid, groups in by_task.items():
            k = task_by_id[kid]
            rho = rho_of(k) if rho_of is not None else min(
                1.0, k.delivered / k.n_images)
            widths = residual_widths(rho)
            for j in range(self.n_seg):
                ent = [(self.li[(g, j)], 1.0) for g in groups]
                if not ent:
                    continue
                add(ent, -np.inf, widths[j])
                for idx, _ in ent:
                    ub_var[idx] = min(ub_var[idx], widths[j])
