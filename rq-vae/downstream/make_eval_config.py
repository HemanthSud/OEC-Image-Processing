"""Generate a predict+metrics-only FLAIR-1 eval config for one downstream
condition (orig / blank / q1 / q2 / q4 / q8 / q16), by patching the shipped
configs/flair-1-config.yaml rather than hand-maintaining a parallel schema.

The shipped classes: block already carries the correct 15-class weighting
for the FLAIR-INC_rgbie_15cl_resnet34-unet checkpoint (confirmed against its
HuggingFace model card), so it is left untouched.

    python3 make_eval_config.py --test-csv csv_recon/val-q8.csv \
        --out-folder outputs/q8 --out eval-q8.yaml
"""
import argparse
import yaml

CKPT = 'checkpoints/unet-r34-rgbie/FLAIR-INC_rgbie_15cl_resnet34-unet_weights.pth'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-config', default='configs/flair-1-config.yaml')
    ap.add_argument('--test-csv', required=True)
    ap.add_argument('--out-folder', required=True)
    ap.add_argument('--out-model-name', default='recon')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    with open(args.base_config) as fh:
        cfg = yaml.safe_load(fh)

    cfg['paths']['out_folder'] = args.out_folder
    cfg['paths']['out_model_name'] = args.out_model_name
    cfg['paths']['test_csv'] = args.test_csv
    cfg['paths']['ckpt_model_path'] = CKPT
    cfg['tasks']['train'] = False
    cfg['tasks']['predict'] = True
    cfg['tasks']['metrics'] = True
    cfg['tasks']['delete_preds'] = True
    cfg['model_framework']['model_provider'] = 'SegmentationModelsPytorch'
    cfg['model_framework']['SegmentationModelsPytorch']['encoder_decoder'] = 'resnet34_unet'
    cfg['use_augmentation'] = False

    with open(args.out, 'w') as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
