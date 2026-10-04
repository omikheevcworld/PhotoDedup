# Photo Dedup — CLIP embeddings, local on the NPU

Companion scripts for the article **"Local AI on NPUs: Building a
zero-cloud photo deduplicator with OpenCLIP"**.

Semantic duplicate-photo finder. CLIP (ViT-B/32) image embeddings run
locally — batched through OpenVINO on an Intel NPU, with a plain CPU
fallback — then grouped by cosine similarity. Nothing leaves the machine.

## The three scripts

| script | what it does |
|---|---|
| `export_ov.py` | one-time: exports the OpenCLIP image encoder to an OpenVINO IR (`models/clip-vitb32-b<N>.xml`). The IR has a *static* batch size, so build one per batch you'll use. |
| `embed.py` | two phases: (1) prepare a 224×224 crop of every photo — ImageMagick subprocesses in parallel, cached as PNGs in `.vitb32-crops/` — (2) run the encoder over the crops in batches. Writes an L2-normalized float32 sidecar (`<photo>.vitb32.npy`) next to each original. |
| `cluster.py` | reads the sidecars, unions every pair with cosine similarity ≥ threshold (transitively, via union-find), and moves each cluster's photos + sidecars into `dup_XXXX/`. Scans non-recursively, so re-running on a sorted folder is a no-op. |

## Setup

```bash
python3.14 -m venv .venv
source .venv/bin/activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install open_clip_torch
pip install openvino
pip install onnx onnxscript     # needed by export_ov.py only
pip install numpy pillow
```

Optional: ImageMagick, for parallel crop preprocessing (without it, `embed.py`
uses a slower in-process Pillow path):

```bash
# Linux:   sudo apt-get install -y imagemagick
# macOS:   brew install imagemagick
# Windows: winget install ImageMagick
```

NPU use also needs the Intel NPU runtime/driver alongside OpenVINO (see the
platform docs for your machine).

## Usage

```bash
# 1. one-time, once per batch size:
python export_ov.py --batch 16

# 2. embed (re-runs skip photos with up-to-date sidecars):
python embed.py ~/photos --backend npu --batch 16 --workers 16
python embed.py ~/photos --backend cpu            # no NPU, or for comparison

# 3. cluster:
python cluster.py ~/photos --dry-run              # report only
python cluster.py ~/photos                        # move duplicates
```

Useful flags: `--force` (re-embed even when sidecars are newer),
`--workers N` (parallel crop workers, default 4), `--threshold 0.92`
(cluster join threshold, default 0.92), `--batch N` (inference batch;
NPU: ≥ 8 recommended — must match an exported IR).

## Notes

- `embed.py` is non-destructive: originals are never modified, only sidecars
  are added. `.vitb32-crops/` is a cache — safe to delete.
- The final line of `embed.py` reports throughput and, when run with `sudo`
  on Intel hardware, average package power for the inference phase
  (Intel RAPL), which is the apples-to-apples window for CPU-vs-NPU
  comparisons.
- Run the same command with `--backend cpu` and `--backend npu` to compare.
- Missing IR for a batch size fails fast with the exact export command to run.
