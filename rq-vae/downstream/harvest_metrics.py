"""Collect FLAIR metrics.json per condition into the quality table the OEC
simulator consumes.

    python3 downstream/harvest_metrics.py \
        --results-root ../FLAIR-1-main/outputs \
        --out results/downstream_summary.json \
        --quality-table ../hypatia_sim/oec_sim/quality_table.json

The simulator reads `quality_table.json` and derives s_q from it. Both
anchorings are written so the choice stays visible and reversible:

  ratio : s_q = mIoU_q / mIoU_ref          -- likely a narrow spread
  floor : s_q = (mIoU_q - mIoU_floor) / (mIoU_ref - mIoU_floor)
          -- anchored at the blanked-RGB condition, i.e. what the segmenter
             gets WITHOUT the delivered image. This is the decision-relevant
             zero and is what UTIL_QUALITY_ANCHOR='floor' selects.

If the spread is still narrow after floor anchoring, that IS the finding --
report it rather than tuning it away.
"""
import argparse
import json
from datetime import date
from pathlib import Path


def read_miou(results_root, cond):
    p = Path(results_root) / cond / 'metrics' / 'metrics.json'
    if not p.is_file():
        return None, None
    with open(p) as fh:
        m = json.load(fh)
    avg = m.get('Avg_metrics')
    miou = float(avg[0]) if isinstance(avg, (list, tuple)) else float(avg)
    if miou > 1.5:                     # FLAIR reports percent
        miou /= 100.0
    return miou, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-root', required=True)
    ap.add_argument('--out', default='results/downstream_summary.json')
    ap.add_argument('--quality-table', required=True)
    ap.add_argument('--depths', default='1,2,4,8,16')
    ap.add_argument('--subset', default='val7050')
    ap.add_argument('--model', default='flair-unet-r34-rgbie')
    args = ap.parse_args()

    depths = [int(d) for d in args.depths.split(',') if d]
    ref, ref_full = read_miou(args.results_root, 'orig')
    floor, _ = read_miou(args.results_root, 'blank')
    if ref is None:
        raise SystemExit("missing the 'orig' condition -- run the uncompressed "
                         "reference first; it is also the checkpoint gate "
                         "(expect mIoU ~= 0.5443).")

    miou, full = {}, {}
    for q in depths:
        v, m = read_miou(args.results_root, f'q{q}')
        if v is not None:
            miou[q], full[f'q{q}'] = v, m

    s_ratio = {q: v / ref for q, v in miou.items()}
    s_floor = ({q: (v - floor) / (ref - floor) for q, v in miou.items()}
               if floor is not None and ref - floor > 1e-9 else None)

    table = {
        'source': f'{args.model} / {args.subset} / {date.today().isoformat()}',
        'miou_ref': round(ref, 6),
        'miou_floor': round(floor, 6) if floor is not None else None,
        'anchor': 'floor' if s_floor else 'ratio',
        'miou': {str(q): round(v, 6) for q, v in sorted(miou.items())},
        's': {str(q): round(v, 6)
              for q, v in sorted((s_floor or s_ratio).items())},
        's_ratio': {str(q): round(v, 6) for q, v in sorted(s_ratio.items())},
        'subset': args.subset,
        'n_images': (ref_full or {}).get('n_images'),
    }
    Path(args.quality_table).parent.mkdir(parents=True, exist_ok=True)
    with open(args.quality_table, 'w') as fh:
        json.dump(table, fh, indent=2, sort_keys=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump({'table': table, 'per_condition': full}, fh, indent=2,
                  sort_keys=True)

    print(f'mIoU_ref  {ref:.4f}   mIoU_floor '
          + (f'{floor:.4f}' if floor is not None else '(not run)'))
    print(f"{'depth':>6s} {'mIoU':>8s} {'s(ratio)':>9s} {'s(floor)':>9s}")
    for q in sorted(miou):
        sf = f'{s_floor[q]:9.4f}' if s_floor else '        -'
        print(f'{q:6d} {miou[q]:8.4f} {s_ratio[q]:9.4f} {sf}')
    lo, hi = min(miou.values()), max(miou.values())
    print(f'\nraw mIoU spread q_min->q_max: {100 * (hi / lo - 1):.1f}%')
    if s_floor:
        lo, hi = min(s_floor.values()), max(s_floor.values())
        print(f'floor-anchored s_q spread   : {100 * (hi / lo - 1):.1f}%  '
              f'(compare: 1-LPIPS gives +12%)')
    print(f'\nwrote {args.quality_table}')


if __name__ == '__main__':
    main()
