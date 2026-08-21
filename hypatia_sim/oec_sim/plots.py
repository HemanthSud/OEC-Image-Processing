"""
Static figures ("screenshots") for the OEC scenario.  Written to
oec_scenario/plots/.  Categorical colors follow one fixed assignment per
scheduler across every panel (validated palette, light surface).
"""

import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from . import config as C

# fixed categorical assignment (entity -> hue, never cycled)
SCHED_COLOR = {
    'mpc':             '#2a78d6',
    'mpc-congestion':  '#0b3d91',
    'mpc-hier':        '#8a4fbf',
    'mpc-oracle':      '#1a1a19',
    'ppo':             '#c77b1a',
    'greedy-adaptive': '#1baf7a',
    'greedy-fixed-1':  '#eda100',
    'greedy-fixed-2':  '#008300',
    'greedy-fixed-4':  '#4a3aa7',
    'greedy-fixed-8':  '#e34948',
    'greedy-fixed-16': '#e87ba4',
}
GRID = dict(color='#e6e6e2', linewidth=0.8)
plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': '#3a3a37',
    'text.color': '#1a1a19', 'xtick.color': '#6f6e66',
    'ytick.color': '#6f6e66', 'font.size': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
})


def _subpoints(topo, t):
    p = topo.sat_pos[t]
    r = np.linalg.norm(p, axis=-1)
    lat = np.degrees(np.arcsin(p[:, 2] / r))
    lon = np.degrees(np.arctan2(p[:, 1], p[:, 0]))
    return lon, lat


def _draw_map(ax, topo, t, show_isls=True, title=None):
    lon, lat = _subpoints(topo, t)
    if show_isls:
        ok = np.flatnonzero(topo.isl_ok[t])
        for l in ok[::3]:                       # sample 1/3 for legibility
            a, b = topo.isl_pairs[l]
            if abs(lon[a] - lon[b]) < 180:      # skip dateline wraps
                ax.plot([lon[a], lon[b]], [lat[a], lat[b]],
                        color='#d3e2f5', lw=0.4, zorder=1)
    ax.scatter(lon, lat, s=1.5, color='#2a78d6', zorder=2)
    for g, gs in enumerate(C.GROUND_STATIONS):
        vis = topo.gsl_ok[t, :, g]
        for s in np.flatnonzero(vis):
            if abs(lon[s] - gs['lon']) < 180:
                ax.plot([lon[s], gs['lon']], [lat[s], gs['lat']],
                        color='#1baf7a', lw=0.8, zorder=3)
        ax.scatter([gs['lon']], [gs['lat']], marker='*', s=90,
                   color='#e34948', zorder=5)
        ax.annotate(gs['name'], (gs['lon'], gs['lat']),
                    textcoords='offset points', xytext=(4, 5), fontsize=7.5,
                    color='#1a1a19')
    for a in C.AOIS:
        ax.scatter([a['lon']], [a['lat']], marker='^', s=36,
                   color='#eda100', zorder=4)
    ax.set_xlim(-180, 180); ax.set_ylim(-75, 75)
    ax.set_xticks(range(-180, 181, 60)); ax.set_yticks(range(-60, 61, 30))
    ax.grid(**GRID)
    ax.set_title(title or f'{C.CONSTELLATION_NAME}  t = {int(topo.times[t])} s',
                 fontsize=10)


def make_all(topo, results):
    out = os.path.join(C.OUT_DIR, 'plots')
    os.makedirs(out, exist_ok=True)

    # 1 — constellation snapshot (big)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    _draw_map(ax, topo, 0,
              title=f'{C.CONSTELLATION_NAME}: {C.N_SATS} satellites, '
                    f'{int(topo.isl_ok[0].sum())} feasible ISLs (sampled), '
                    f'GSLs in green, AOIs ▲, GBS ★   (t = 0 s)')
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'constellation_map.png'), dpi=150)
    plt.close(fig)

    # 2 — three time snapshots
    idx = [0, C.N_SLOTS // 3, 2 * C.N_SLOTS // 3]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.4))
    for ax, t in zip(axes, idx):
        _draw_map(ax, topo, t, show_isls=False)
    fig.suptitle('Sub-satellite points and active ground links over the run',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'snapshots.png'), dpi=150)
    plt.close(fig)

    # 3 — contact Gantt (first 2 h)
    from .topology import contact_windows
    wins = [w for w in contact_windows(topo) if w['start_s'] <= 7200]
    fig, ax = plt.subplots(figsize=(11, 3.2))
    names = [g['name'] for g in C.GROUND_STATIONS]
    for w in wins:
        y = names.index(w['gs'])
        ax.barh(y, max(w['duration_s'], 40) / 60, left=w['start_s'] / 60,
                height=0.55, color='#2a78d6', edgecolor='white', linewidth=0.6)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel('time (min)')
    ax.set_xlim(0, 120)
    ax.grid(axis='x', **GRID)
    ax.set_title(f'Satellite contact windows per GBS, first 2 h '
                 f'(min elevation {C.MIN_ELEV_DEG:.0f}°) — '
                 f'coverage is near-continuous with {C.N_SATS} satellites',
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'contact_gantt.png'), dpi=150)
    plt.close(fig)

    # 4 — ISL distance dynamics (time-varying inter-plane links)
    intra = np.array([a // C.SATS_PER_PLANE == b // C.SATS_PER_PLANE
                      for a, b in topo.isl_pairs])
    fig, ax = plt.subplots(figsize=(9, 3.2))
    tt = topo.times / 60
    inter_d = topo.isl_dist[:, ~intra]
    ax.fill_between(tt, inter_d.min(axis=1), inter_d.max(axis=1),
                    color='#d3e2f5', label='inter-plane min–max')
    ax.plot(tt, inter_d.mean(axis=1), color='#2a78d6', lw=2,
            label='inter-plane mean')
    ax.plot(tt, topo.isl_dist[:, intra].mean(axis=1), color='#1baf7a', lw=2,
            label='intra-plane (constant)')
    ax.set_xlabel('time (min)'); ax.set_ylabel('ISL length (km)')
    ax.grid(**GRID); ax.legend(frameon=False, fontsize=8)
    ax.set_title('ISL lengths: all links satisfy line-of-sight '
                 f'(graze > {C.LOS_GRAZE_KM:.0f} km) and range '
                 f'≤ {C.ISL_MAX_KM:.0f} km', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'isl_dynamics.png'), dpi=150)
    plt.close(fig)

    # 5 — scheduler comparison (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for name, res in results.items():
        h, col = res['hist'], SCHED_COLOR.get(name, '#6f6e66')
        tt = np.array(h['t_s']) / 60
        lw = 2.4 if name == 'mpc' else 1.6
        axes[0, 0].plot(tt, np.array(h['delivered_images']) / 1e6,
                        color=col, lw=lw, label=name)
        axes[0, 1].plot(tt, h['utility'], color=col, lw=lw, label=name)
        axes[1, 0].plot(tt, np.array(h['backlog_bits']) / 1e9,
                        color=col, lw=lw, label=name)
    axes[0, 0].set_title('cumulative images delivered (millions)', fontsize=10)
    axes[0, 1].set_title('cumulative weighted utility '
                         r'$\sum_k w_k\,\phi_k\,u_{q_k}\,\rho_k$', fontsize=10)
    axes[1, 0].set_title('compressed backlog in orbit (Gbit)', fontsize=10)
    for ax in axes.flat[:3]:
        ax.set_xlabel('time (min)'); ax.grid(**GRID)
    axes[0, 1].legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    scheds = list(results)
    bottoms = np.zeros(len(scheds))
    shades = ['#d3e2f5', '#8db8e8', '#4a90dd', '#1d5eb0']
    depth_shade = {q: shades[i] for i, q in enumerate(sorted(C.DEPTHS))}
    for q in C.DEPTHS:
        vals = [sum(1 for k in results[s]['tasks'] if k.depth == q)
                for s in scheds]
        ax.bar(range(len(scheds)), vals, bottom=bottoms,
               color=depth_shade[q], edgecolor='white', linewidth=1,
               label=f'depth {q}')
        bottoms += vals
    ax.set_xticks(range(len(scheds)),
                  [s.replace('greedy-', 'g-') for s in scheds],
                  rotation=20, fontsize=8)
    ax.set_title('chosen compression depth per task', fontsize=10)
    ax.grid(axis='y', **GRID)
    ax.set_ylim(0, len(next(iter(results.values()))['tasks']) * 1.22)
    ax.legend(frameon=False, fontsize=8, ncols=4, loc='upper center')
    fig.suptitle('MPC vs greedy baselines — maximize delivered images / utility',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'results_comparison.png'), dpi=150)
    plt.close(fig)

    # 6 — completion-delay CDF per scheduler (timeliness: Xuanhao's ask)
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, res in results.items():
        col = SCHED_COLOR.get(name, '#6f6e66')
        delays = sorted(k.completion_slot * C.SLOT_S - k.arrival_slot * C.SLOT_S
                        for k in res['tasks'] if k.completion_slot is not None)
        if not delays:
            continue
        yy = np.arange(1, len(delays) + 1) / len(res['tasks'])
        lw = 2.4 if name == 'mpc' else 1.6
        ax.step(np.array(delays) / 60, yy, where='post', color=col, lw=lw,
               label=name)
    ax.set_xlabel('completion delay (min)')
    ax.set_ylabel('fraction of tasks completed')
    ax.set_title('Task completion-delay CDF (unfinished tasks excluded)',
                 fontsize=10)
    ax.grid(**GRID); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'delay_cdf.png'), dpi=150)
    plt.close(fig)

    print(f'  wrote 6 figures -> {os.path.relpath(out)}/')
