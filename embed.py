#!/usr/bin/env python3
"""Generate CLIP image embeddings for a folder of photos (no model files).

For each image it writes an L2-normalized float32 sidecar
`<name>.vitb32.npy` next to the original.  Existing, up-to-date sidecars
are skipped, so running the script twice is cheap.

The work happens in two phases:
  1. prepare a 224x224 crop of every photo — ImageMagick subprocesses,
     run in parallel (--workers), or an in-process Pillow transform when
     ImageMagick is missing; crops are cached in .vitb32-crops/
  2. run the encoder over the crops in batches of --batch

Inference backend is selected with --backend:
  cpu   -- run the OpenCLIP PyTorch model on the CPU (default)
  npu   -- run an OpenVINO IR build of the same encoder on the Intel NPU
           (one IR per batch size; run export_ov.py --batch <N> first)

The final "avg power" figure covers phase 2 only, so CPU vs NPU compares
inference alone (needs sudo, Intel RAPL).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms as T
from PIL import Image

import open_clip

IM_CMD = shutil.which("magick") or shutil.which("convert")

MODEL_NAME = "ViT-B-32-quickgelu"
PRETRAINED = "openai"
SUFFIX = ".vitb32"
EXTS = frozenset({".jpg", ".jpeg", ".png", ".webp"})
HERE = Path(__file__).resolve().parent

# CLIP's fixed per-channel mean/std for its 224x224 input
# (OpenAI's published values).
CLIP_MEAN = [0.481454, 0.457827, 0.408215]
CLIP_STD = [0.268629, 0.261302, 0.275777]


def crop_photo(p: Path, cache_dir: Path) -> Path:
    """Phase 1: a 224x224 PNG crop of the photo, written to the cache.
    Uses an ImageMagick subprocess (runs outside the GIL) when available,
    otherwise an in-process Pillow transform."""
    out = cache_dir / f"{p.name}.png"
    if out.is_file() and out.stat().st_mtime >= p.stat().st_mtime:
        return out
    if IM_CMD:
        raw = subprocess.run(
            [IM_CMD, str(p),
             "-filter", "Cubic",
             "-resize", "256x256^",
             "-gravity", "center",
             "-extent", "224x224",
             "png:-"],
            capture_output=True, check=True, timeout=300).stdout
        out.write_bytes(raw)
    else:
        T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BICUBIC,
                            antialias=True),
                   T.CenterCrop(224)])(Image.open(p).convert("RGB")).save(out)
    return out


def normalize_crop(im: Image.Image) -> torch.Tensor:
    """Normalize a ready 224x224 crop (phase 2)."""
    x = T.functional.to_tensor(im)
    mean = torch.tensor(CLIP_MEAN, dtype=torch.float32).reshape(3, 1, 1)
    std = torch.tensor(CLIP_STD, dtype=torch.float32).reshape(3, 1, 1)
    return x.sub_(mean).div_(std)


def package_energy_uj() -> float | None:
    """CPU package energy from Intel RAPL, in microjoules.
    None when unavailable (non-Intel parts, or no permission)."""
    try:
        return float(Path("/sys/class/powercap/intel-rapl:0/energy_uj").read_text())
    except OSError:
        return None


def model_path(batch: int) -> Path:
    """IR with a static batch dimension equal to `batch`.

    The ONNX exporter bakes batch-derived constants into internal Reshape
    nodes, so each batch size needs its own IR (export_ov.py --batch N).

    """
    p = HERE / "models" / f"clip-vitb32-b{batch}.xml"
    if not p.exists():
        raise SystemExit(f"{p} not found; build it with: "
                         f"python export_ov.py --batch {batch}")
    return p


def normalize_batch(vecs) -> np.ndarray:
    vecs = np.asarray(vecs, dtype=np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True).astype(np.float32)


def list_photos(directory: Path) -> list[Path]:
    return sorted(
        p for p in directory.iterdir()
        if p.suffix.lower() in EXTS and p.is_file())


def make_npu(batch: int):
    import openvino as ov

    core = ov.Core()
    if "NPU" not in core.available_devices:
        raise SystemExit("NPU not in OpenVINO device list; install the NPU stack first")
    ir = model_path(batch)
    hint = "THROUGHPUT" if batch > 1 else "LATENCY"
    model = core.read_model(str(ir))
    model.reshape([batch, 3, 224, 224])
    compiled = core.compile_model(model, "NPU",
                                  config={"PERFORMANCE_HINT": hint})
    print(f"backend: NPU (compiled {ir.name}, hint={hint}, "
          f"batch={batch})", flush=True)
    return compiled.create_infer_request()


def req_infer(req, tensor: torch.Tensor) -> np.ndarray:
    # OpenVINO keys a model's inputs by index; this one has a single
    # input (index 0), so the result likewise comes back first.
    return np.asarray(req.infer({0: tensor.numpy()})[0], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", default=".",
                        help="folder to scan (default: current directory)")
    parser.add_argument("--force", action="store_true",
                        help="re-embed even if a sidecar is newer than the image")
    parser.add_argument("--backend", choices=["cpu", "npu"], default="cpu",
                        help="inference backend (default: cpu)")
    parser.add_argument("--batch", type=int, default=1,
                        help="inference batch size (NPU: >= 8 recommended)")
    parser.add_argument("--workers", type=int, default=4,
                        help="parallel crop workers (default 4)")
    args = parser.parse_args()
    batch = max(1, args.batch)

    directory = Path(args.directory)
    photos = list_photos(directory)
    if not photos:
        raise SystemExit(f"no photos found in {directory.resolve()}")

    model, _, _ = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED)
    model.eval()

    req = None
    if args.backend == "npu":
        model_path(batch)
        req = make_npu(batch)

    # Warm the backend so the first real chunk is not slower than the
    # rest: one dummy forward through the same shapes.
    with torch.inference_mode():
        dummy = torch.rand(batch, 3, 224, 224)
        if args.backend == "npu":
            req_infer(req, dummy)
        else:
            model.encode_image(dummy)

    pending = []
    skipped = 0
    for p in photos:
        sidecar = p.with_name(p.stem + SUFFIX + ".npy")
        if not args.force and sidecar.exists() \
                and sidecar.stat().st_mtime >= p.stat().st_mtime:
            skipped += 1
            continue
        pending.append((p, sidecar))

    # Phase 1: parallel decode + resize, crops cached as 224x224 PNGs.
    cache_dir = directory / ".vitb32-crops"
    if IM_CMD:
        print(f"preprocessing: {IM_CMD} (parallel, {args.workers} "
              f"workers)", flush=True)
    else:
        print("preprocessing: in-process Pillow (ImageMagick not found; "
              "install for parallel preprocessing)", flush=True)
    t0_prep = time.perf_counter()
    if pending:
        cache_dir.mkdir(exist_ok=True)
        with ThreadPoolExecutor(max_workers=args.workers,
                                thread_name_prefix="crop") as pool:
            crops = list(pool.map(lambda p: crop_photo(p[0], cache_dir), pending))
    else:
        crops = []
    print(f"prepared {len(crops)} crop(s) in "
          f"{time.perf_counter() - t0_prep:.1f}s", flush=True)

    # Phase 2: batched inference over the crops.
    e0 = package_energy_uj()
    t0 = time.perf_counter()
    embedded = 0
    with ThreadPoolExecutor(max_workers=args.workers,
                            thread_name_prefix="prep") as pool:
        # Load one chunk ahead so decoding overlaps the inference.
        futs = [pool.submit(lambda c: normalize_crop(Image.open(c)),
                            c) for c in crops[:batch]]
        for i in range(0, len(crops), batch):
            chunk = crops[i:i + batch]
            nxt_futs = [pool.submit(lambda c: normalize_crop(Image.open(c)),
                                    c) for c in crops[i + batch:i + 2 * batch]]
            # 1. Collect the loaded tensors for this chunk
            tensors = torch.stack([f.result() for f in futs])
            # 2. Pad the last short chunk to the static batch size
            #    (the IR was built for exactly `batch` inputs)
            if len(chunk) < batch:
                tensors = torch.cat([tensors,
                                     torch.zeros(batch - len(chunk), 3, 224, 224)])
            n = len(chunk)
            t1 = time.perf_counter()
            # 3. Run inference
            if args.backend == "npu":
                vecs = req_infer(req, tensors)
            else:
                with torch.inference_mode():
                    vecs = model.encode_image(tensors).numpy()
            ms = (time.perf_counter() - t1) * 1000.0 / n
            # 4. L2-normalize, dropping any padded rows
            vecs = normalize_batch(vecs)[:n]
            # 5. Save each photo's sidecar
            for j, (p, sidecar) in enumerate(pending[i:i + n]):
                np.save(sidecar, vecs[j])
                embedded += 1
                print(f"[{i + j + 1}/{len(photos)}] {p.name} "
                      f"{ms:8.1f} ms -> {sidecar.name}", flush=True)
            futs = nxt_futs
    total = time.perf_counter() - t0
    summary = (f"embedded {embedded} in {total:.1f}s "
               f"({embedded / max(total, 1e-09):.1f} img/s) "
               f"[backend={args.backend}, batch={batch}], "
               f"{skipped} skipped (cached)")
    e1 = package_energy_uj()
    if e0 is not None and e1 is not None and total > 0:
        summary += f", avg power {(e1 - e0) / 1e6 / total:.1f} W"
    print(summary, flush=True)


if __name__ == "__main__":
    main()
