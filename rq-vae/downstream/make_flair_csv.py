"""Turn a reconstruction root into the 2-column headerless CSV FLAIR expects.

Column 0 (image) is remapped into the reconstruction tree; column 1 (mask) is
left pointing at the original labels -- the ground truth never changes.

    python3 downstream/make_flair_csv.py --recon-root /scratch/flair_recon_q8 \
        --base-csv downstream/flair-1-paths-val-7050.csv \
        --out ../FLAIR-1-main/csv_recon/val-q8.csv
"""
import argparse
import csv
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-csv', required=True)
    ap.add_argument('--recon-root', required=True,
                    help="reconstruction tree, or 'orig' to keep originals")
    ap.add_argument('--data-root', default='..')
    ap.add_argument('--out', required=True)
    ap.add_argument('--allow-missing', type=int, default=0)
    args = ap.parse_args()

    root = Path(args.data_root).resolve()
    rows, missing = [], []
    with open(args.base_csv) as fh:
        for rec in csv.reader(fh):
            if len(rec) < 2:
                continue
            img, msk = rec[0], rec[1]
            if args.recon_root != 'orig':
                src = (root / img).resolve() if not Path(img).is_absolute() \
                    else Path(img).resolve()
                try:
                    rel = src.relative_to(root)
                except ValueError:
                    rel = Path(*src.parts[-4:])
                new = Path(args.recon_root) / rel
                if not new.is_file():
                    missing.append(str(new))
                img = str(new)
            rows.append([img, msk])

    if missing and len(missing) > args.allow_missing:
        raise SystemExit(
            f'{len(missing)} reconstructions missing, e.g. {missing[:3]}. '
            f'Re-run recon_to_geotiff.py or pass --allow-missing.')

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', newline='') as fh:
        csv.writer(fh).writerows(rows)
    print(f'{len(rows)} rows -> {args.out}'
          + (f'  ({len(missing)} missing, allowed)' if missing else ''))


if __name__ == '__main__':
    main()
