"""
Profile RQ-VAE encoder inference time (computing delay at the RQ satellite node).

Run on the server after training:
    python3 profile_compute_delay.py
"""
import os
import sys
import time
import yaml
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'rq-vae'))

from rqvae.models.rqvae.rqvae import RQVAE

OUTPUT_DIRS = [
    "../rq-vae/output/flair-rqvae-8x8x1",
    "../rq-vae/output/flair-rqvae-8x8x8",
]

N_WARMUP = 10
N_RUNS   = 100
IMG_SIZE = 512
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_encoder(output_dir):
    config_path = os.path.join(output_dir, "config.yaml")
    ckpt_path   = os.path.join(output_dir, "best_model.pt")
    with open(config_path) as f:
        config = OmegaConf.create(yaml.load(f, Loader=yaml.FullLoader))
    hps      = dict(config.arch.hparams)
    ddconfig = dict(config.arch.ddconfig)
    model = RQVAE(**hps, ddconfig=ddconfig, checkpointing=False).to(DEVICE)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, config


def profile_encoder(model):
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
    # Warmup
    with torch.no_grad():
        for _ in range(N_WARMUP):
            _ = model.encode(dummy)
    torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(N_RUNS):
            start = time.perf_counter()
            _ = model.encode(dummy)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)  # ms

    mean_ms = sum(times) / len(times)
    std_ms  = (sum((t - mean_ms) ** 2 for t in times) / len(times)) ** 0.5
    return mean_ms, std_ms


def main():
    print("=== RQ Node Computing Delay Profile ===\n")
    print(f"{'Model':<30} {'Mean (ms)':>10} {'Std (ms)':>10} {'Delay (ns)':>14}")
    print("-" * 68)

    results = {}
    for out_dir in OUTPUT_DIRS:
        name = os.path.basename(out_dir)
        if not os.path.isdir(out_dir):
            print(f"{name:<30} SKIPPED (not found)")
            continue
        model, _ = load_encoder(out_dir)
        mean_ms, std_ms = profile_encoder(model)
        delay_ns = int(mean_ms * 1e6)
        print(f"{name:<30} {mean_ms:>10.3f} {std_ms:>10.3f} {delay_ns:>14}")
        results[name] = delay_ns
        del model
        torch.cuda.empty_cache()

    # Write for use by generate_sim_inputs.py
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compute_delays.txt")
    with open(out_path, "w") as f:
        for name, ns in results.items():
            f.write(f"{name},{ns}\n")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
