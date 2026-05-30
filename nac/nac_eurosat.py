"""
NAC (N-gram Arithmetic Coding) for RQ-VAE codes.

This script:
  1. Reads RQ-VAE generated codes (H×W×D flattened per image)
  2. Builds an N-gram frequency table from training codes
  3. Encodes/decodes test codes using arithmetic coding
  4. Reports compression rates and timing

Usage:
    cd nac/
    python nac_eurosat.py --dataset flair --height 8 --width 8 --depth 4 --image-size 512
"""
from ngram import NGramModel
from arithmetic_coding import ArithmeticEncoder
import argparse
import sys
import logging
import os
import time


def readcode(filename, n=None):
    result = []
    with open(filename, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if n is not None and i >= n:
                break
            line = line.strip()
            if line:
                numbers = [int(x) for x in line.split()]
                result.append(numbers)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description='NAC for flattened RQ-VAE codes')
    parser.add_argument('--dataset', default='eurosat',
                        help='Dataset name used for logs/model names')
    parser.add_argument('--height', type=int, default=8,
                        help='Latent code grid height')
    parser.add_argument('--width', type=int, default=8,
                        help='Latent code grid width')
    parser.add_argument('--depth', type=int, default=4,
                        help='RQ depth')
    parser.add_argument('--n-embed', type=int, default=2048,
                        help='Codebook size')
    parser.add_argument('--ngram', type=int, default=2,
                        help='N-gram order')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Additive smoothing constant')
    parser.add_argument('--n-train', type=int, default=900,
                        help='Number of code sequences used to fit NAC')
    parser.add_argument('--n-total', type=int, default=1000,
                        help='Number of total sequences read before test slicing')
    parser.add_argument('--image-size', type=int, default=64,
                        help='Original square image size in pixels')
    parser.add_argument('--channels', type=int, default=3,
                        help='Original image channels')
    parser.add_argument('--bits-per-pixel-channel', type=int, default=8,
                        help='Raw bits per image channel')
    parser.add_argument('--code-file', default=None,
                        help='Optional path to codes text file')
    return parser.parse_args()


args = parse_args()

N = args.ngram
K = args.smoothing
D = args.depth
H, W = args.height, args.width
N_TRAIN = args.n_train
N_TOTAL = args.n_total
BITS_PER_CODE = (args.n_embed - 1).bit_length()
RAW_IMAGE_BITS = args.image_size * args.image_size * args.channels * args.bits_per_pixel_channel


logger = logging.getLogger()
logger.setLevel(logging.INFO)

os.makedirs("logs", exist_ok=True)

logfile = f"logs/{N}gram_{args.dataset}_{H}x{W}x{D}_log.txt"
print("Log:", logfile)
file_handler = logging.FileHandler(logfile, mode='w', encoding='utf-8')
console_handler = logging.StreamHandler(sys.stdout)

formatter = logging.Formatter('%(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

if args.code_file is not None:
    filename = args.code_file
elif args.dataset == 'eurosat':
    filename = f"data/codes{H}x{W}x{D}.txt"
else:
    filename = f"data/{args.dataset}_codes{H}x{W}x{D}.txt"

logger.info(f"N={N}, K={K}")
logger.info(f"Code shape: {H}×{W}×{D} = {H*W*D} codes per image")
logger.info(f"Data: {filename}")
logger.info(f"Raw image bits: {RAW_IMAGE_BITS:,}")
logger.info(f"Bits per code: {BITS_PER_CODE}")

training_sequences = readcode(filename, N_TRAIN)

model = NGramModel(n=N, k=K, start_token=-1, end_token=-2)
model.fit(training_sequences)

os.makedirs("models", exist_ok=True)
model.save(f"models/{N}gram_{args.dataset}_{H}x{W}x{D}.pkl")

encoder = ArithmeticEncoder(ngram_model=model, bits=32)
print("Encoder Created")

codes = readcode(filename, N_TOTAL)[N_TRAIN:N_TOTAL]

avgrate = []
encode_times_ms = []
decode_times_ms = []

for i in range(len(codes)):
    logger.info(f"Code: {i}")
    test_sequence = codes[i]

    t0 = time.perf_counter()
    encoded_bits = encoder.encode(test_sequence)
    t1 = time.perf_counter()
    encode_ms = (t1 - t0) * 1000.0
    encode_times_ms.append(encode_ms)

    logger.info(
        f"Encoded: {len(encoded_bits)} bits, encode_time={encode_ms:.4f} ms")
    rate = len(encoded_bits) / (len(test_sequence) * BITS_PER_CODE)
    avgrate.append(rate)
    logger.info(f"Compression Rate: {rate:.2%}")

    compression_ratio = RAW_IMAGE_BITS / len(encoded_bits)
    logger.info(f"vs Raw Image Compression Ratio: {compression_ratio:.1f}×")

    t0 = time.perf_counter()
    decoded_sequence = encoder.decode(encoded_bits)
    t1 = time.perf_counter()
    decode_ms = (t1 - t0) * 1000.0
    decode_times_ms.append(decode_ms)

    logger.info(f"Decode_time={decode_ms:.4f} ms")
    logger.info(
        f"Verification: {'Correct' if decoded_sequence == test_sequence else 'Wrong'}")

logger.info("")
logger.info("=" * 60)
logger.info("SUMMARY")
logger.info("=" * 60)
logger.info(
    f"Average Compression Rate (vs uncompressed codes): {sum(avgrate)/len(avgrate):.2%}")
avg_raw_ratio = sum(
    RAW_IMAGE_BITS / (len(encoder.encode(codes[i])))
    for i in range(len(codes))
) / len(codes)
logger.info(f"Average Compression Ratio (vs raw RGB): {avg_raw_ratio:.1f}×")
logger.info("")
logger.info("Timing summary (ms):")
avg_encode = sum(encode_times_ms) / len(encode_times_ms)
avg_decode = sum(decode_times_ms) / len(decode_times_ms)
logger.info(f"  Average encode time: {avg_encode:.4f} ms")
logger.info(f"  Average decode time: {avg_decode:.4f} ms")
logger.info(
    f"  encode: min={min(encode_times_ms):.4f}, max={max(encode_times_ms):.4f}")
logger.info(
    f"  decode: min={min(decode_times_ms):.4f}, max={max(decode_times_ms):.4f}")
