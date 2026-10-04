#!/usr/bin/env python3
"""One-time export of the OpenCLIP image encoder to an OpenVINO IR.

Produces models/clip-vitb32-b<B>.xml(+.bin) with a *static* batch of B.
The NPU wants a fully concrete shape, so one IR is built per batch size
(the ONNX exporter bakes batch-derived constants into internal Reshape
nodes, which a dynamically-batched IR gets wrong for B > 1).

Usage: python export_ov.py --batch 8
Re-running for an existing file is a no-op unless --force is given.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
MODEL_NAME = "ViT-B-32-quickgelu"
PRETRAINED = "openai"


def ir_path(batch: int) -> Path:
    return HERE / "models" / f"clip-vitb32-b{batch}.xml"


def main() -> None:
    import openvino as ov

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1,
                        help="static batch size to bake into the model (default 1)")
    parser.add_argument("--force", action="store_true",
                        help="re-export even if the IR already exists")
    args = parser.parse_args()
    batch = max(1, args.batch)
    ir = ir_path(batch)

    if not ir.exists() or args.force:
        import open_clip

        model, _, _ = open_clip.create_model_and_transforms(
            MODEL_NAME, pretrained=PRETRAINED)
        visual = model.visual.eval()
        example = torch.randn(batch, 3, 224, 224)
        ir.parent.mkdir(parents=True, exist_ok=True)
        print(f"exporting image encoder to OpenVINO IR "
              f"(static batch={batch}) ...", flush=True)
        onnx_path = ir.with_suffix(".onnx")
        try:
            torch.onnx.export(visual, (example,), str(onnx_path),
                              opset_version=17,
                              input_names=["images"],
                              output_names=["embeddings"])
        except ModuleNotFoundError as exc:
            onnx_path.unlink(missing_ok=True)
            raise SystemExit(
                f"export needs the ONNX toolchain (missing: {exc.name}). "
                f"Run: pip install onnx onnxscript")
        ov_model = ov.convert_model(str(onnx_path))
        ov.save_model(ov_model, str(ir))
        onnx_path.unlink(missing_ok=True)          # drop the ~300 MB ONNX
        ir.with_suffix(".onnx.data").unlink(missing_ok=True)  # intermediate
        del model

    ir_model = ov.Core().read_model(str(ir))
    print(f"IR: {ir}")
    for p in ir_model.inputs:
        print("input :", str(p.get_partial_shape()), p.get_element_type())
    for o in ir_model.outputs:
        print("output:", str(o.get_partial_shape()), o.get_element_type())


if __name__ == "__main__":
    main()
