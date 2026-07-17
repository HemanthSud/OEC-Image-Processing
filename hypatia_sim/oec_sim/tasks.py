"""
Task generation K: a task arrives when a satellite passes over an area of
interest (elevation >= AOI_ELEV_DEG) and the AOI's cooldown has elapsed.

Task k: source satellite s_k (the imaging satellite), destination GBS g_k
(nearest GBS to the AOI), arrival slot a_k, N_k images, weight w_k, soft
deadline d_k.  Matches the task model in OEC_RQ_NAC.pdf Sec. 1.1.
"""

from dataclasses import dataclass, field

import numpy as np

from . import config as C
from . import geometry as G


@dataclass
class Task:
    kid: int
    aoi: str
    src_sat: int
    dst_gs: int
    arrival_slot: int
    n_images: int
    weight: int
    deadline_s: int              # absolute sim time of the soft deadline
    # scheduler state
    depth: int = None            # committed q_k (None until chosen)
    delivered: float = 0.0       # images fully delivered r_k
    delivered_utility: float = 0.0
    delivery_slots: dict = field(default_factory=dict)  # slot -> images

    def encoded_by(self, t_s):
        """Images encoded by absolute time t_s (encode starts at arrival)."""
        avail = (t_s - self.arrival_slot * C.SLOT_S) * C.ENC_IMGS_PER_S
        return float(np.clip(avail, 0.0, self.n_images))

    def freshness(self, t_s):
        """phi_k(t): 1 up to the soft deadline, exponential decay after."""
        if t_s <= self.deadline_s:
            return 1.0
        return float(np.exp(-C.FRESHNESS_ALPHA * (t_s - self.deadline_s)))


def _nearest_gs(aoi):
    p = G.latlon_ecef(aoi['lat'], aoi['lon'])
    gs = G.latlon_ecef([g['lat'] for g in C.GROUND_STATIONS],
                       [g['lon'] for g in C.GROUND_STATIONS])
    return int(np.argmin(np.linalg.norm(gs - p, axis=-1)))


def generate_tasks(topo):
    rng = np.random.default_rng(C.RNG_SEED)
    cooldown_slots = C.AOI_COOLDOWN_S // C.SLOT_S
    tasks = []
    next_ok = np.zeros(len(C.AOIS), dtype=int)     # earliest slot per AOI

    for t in range(C.N_SLOTS):
        for a, aoi in enumerate(C.AOIS):
            if t < next_ok[a] or len(tasks) >= C.MAX_TASKS:
                continue
            over = np.flatnonzero(topo.aoi_elev[t, :, a] >= C.AOI_ELEV_DEG)
            if len(over) == 0:
                continue
            src = int(over[np.argmax(topo.aoi_elev[t, over, a])])
            tasks.append(Task(
                kid=len(tasks), aoi=aoi['name'], src_sat=src,
                dst_gs=_nearest_gs(aoi), arrival_slot=t,
                n_images=int(rng.integers(C.TASK_IMAGES_MIN,
                                          C.TASK_IMAGES_MAX + 1)),
                weight=int(rng.choice(C.TASK_WEIGHTS)),
                deadline_s=int(t * C.SLOT_S
                               + rng.integers(C.TASK_DEADLINE_MIN_S,
                                              C.TASK_DEADLINE_MAX_S + 1)),
            ))
            next_ok[a] = t + cooldown_slots
    return tasks
