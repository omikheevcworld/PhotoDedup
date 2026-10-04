#!/usr/bin/env python3
"""Group CLIP-duplicate photos into per-cluster folders.

Reads the `.vitb32.npy` sidecars written by embed.py, unions every
pair with cosine similarity >= --threshold into a cluster, and moves
each cluster's photos (with their sidecars) into its own
`dup_XXXX/` subfolder.  Photos with no near-duplicate stay put.

The scan is non-recursive, so re-running is a no-op: clustered photos
already live in subfolders.  Run this after embed.py.

Usage:
    python cluster.py photos
    python cluster.py photos --dry-run
"""
from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

SUFFIX = ".vitb32"
EXTS = frozenset({".jpg", ".jpeg", ".png", ".webp"})


def list_photos(directory: Path) -> list[Path]:
    return sorted(
        p for p in directory.iterdir()
        if p.suffix.lower() in EXTS and p.is_file())


def load_embeddings(photos: list[Path]) -> tuple[list[Path], np.ndarray]:
    embedded = [
        p for p in photos
        if p.with_name(p.stem + SUFFIX + ".npy").is_file()]
    vecs = np.stack([np.load(p.with_name(p.stem + SUFFIX + ".npy"))
                     for p in embedded]).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return embedded, vecs


def clusters_of(vecs: np.ndarray, threshold: float) -> list[list[int]]:
    """Indices grouped so that each group reaches the start through
    >= threshold cosine hops (union-find over the similarity matrix).

    """
    n = len(vecs)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    hits = np.argwhere(np.triu(vecs @ vecs.T, k=1) >= threshold)
    for i, j in hits.tolist():
        ri, rj = find(int(i)), find(int(j))
        if ri != rj:
            parent[ri] = rj

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return [m for m in groups.values() if len(m) >= 2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", default=".",
                        help="folder to cluster (default: current directory)")
    parser.add_argument("--threshold", type=float, default=0.92,
                        help="cosine similarity that joins two photos (default 0.92)")
    parser.add_argument("--dry-run", action="store_true",
                        help="report clusters without moving anything")
    args = parser.parse_args()

    directory = Path(args.directory).resolve()
    photos = list_photos(directory)
    if not photos:
        raise SystemExit(f"no photos found in {directory}")

    embedded, vecs = load_embeddings(photos)
    skipped = len(photos) - len(embedded)
    if skipped:
        print(f"warning: {skipped} photo(s) have no {SUFFIX}"
              f".npy sidecar and are ignored (run embed.py first)")

    if len(embedded) < 2:
        raise SystemExit("need at least 2 embedded photos to cluster")

    clusters = clusters_of(vecs, args.threshold)
    if not clusters:
        print(f"no duplicate candidates (threshold {args.threshold}); nothing to do")
        return
    clusters.sort(key=lambda m: min(embedded[i].name for i in m))

    moved = 0
    print(f"{len(clusters)} cluster(s), "
          f"{sum(len(m) for m in clusters)} photo(s), threshold "
          f"{args.threshold}:")
    for c, members in enumerate(clusters, start=1):
        folder = directory / f"dup_{c:04d}"
        print(f"  {folder.name}/ ({len(members)})")
        for i in sorted(members, key=lambda j: embedded[j].name):
            img = embedded[i]
            sidecar = img.with_name(img.stem + SUFFIX + ".npy")
            print(f"    {img.name}")
            if args.dry_run:
                continue
            folder.mkdir(exist_ok=True)
            shutil.move(str(img), folder / img.name)
            if sidecar.is_file():
                shutil.move(str(sidecar), folder / sidecar.name)
            moved += 1

    verb = "would move" if args.dry_run else "moved"
    print(f"{verb} {moved} photo(s) into {len(clusters)} cluster folder(s)")


if __name__ == "__main__":
    main()
