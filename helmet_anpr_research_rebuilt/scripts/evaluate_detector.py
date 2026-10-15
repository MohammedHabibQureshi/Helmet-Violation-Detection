#!/usr/bin/env python3
"""CUDA / GPU inference test for the Helmet Violation Detection project.

Verifies:
  - torch.cuda.is_available()
  - device name and memory
  - a real tensor operation on CUDA
  - model loading on CUDA
  - one real inference pass on CUDA
  - GPU sync and timing
"""

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    results = {"checks": [], "all_passed": True}

    def check(name, fn):
        try:
            detail = fn()
            results["checks"].append({
                "name": name, "passed": True, "detail": detail})
            print(f"[PASS] {name}: {detail}")
        except Exception as exc:  # noqa: BLE001
            results["all_passed"] = False
            results["checks"].append({
                "name": name, "passed": False, "error": str(exc)})
            print(f"[FAIL] {name}: {exc}")

    import torch

    def c1():
        assert torch.cuda.is_available(), "CUDA not available"
        return f"torch.cuda.is_available()=True, device_count={torch.cuda.device_count()}"

    def c2():
        name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        return (f"name={name}, capability=({props.major},{props.minor}), "
                f"total_mem={props.total_memory/1e9:.2f}GB")

    def c3():
        torch.cuda.set_device(0)
        a = torch.randn(2048, 2048, device="cuda")
        b = torch.randn(2048, 2048, device="cuda")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            c = a @ b
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 10 * 1000.0
        assert c.shape == (2048, 2048)
        return f"matmul_2048 mean={ms:.3f} ms, out={tuple(c.shape)}"

    def c4():
        from ultralytics import YOLO
        model = YOLO("yolov8s.pt")
        model.to("cuda")
        return f"YOLOv8s moved to {model.device}"

    def c5():
        import numpy as np
        from ultralytics import YOLO
        model = YOLO("yolov8s.pt")
        model.to("cuda")
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        res = model.predict(frame, device="cuda", verbose=False)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0
        n = 0
        for r in res:
            if r.boxes is not None:
                n = len(r.boxes)
        return (f"inference on 640x640 blank took {ms:.1f} ms, "
                f"{n} detections (blank frame)")

    check("cuda_available", c1)
    check("device_name_memory", c2)
    check("tensor_op_cuda", c3)
    check("model_load_cuda", c4)
    check("inference_pass_cuda", c5)

    out = PROJECT_ROOT / "reports" / "cuda_diagnostics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)

    md = PROJECT_ROOT / "reports" / "CUDA_DIAGNOSTICS.md"
    with md.open("w", encoding="utf-8") as fh:
        fh.write("# CUDA Diagnostics\n\n")
        fh.write(f"Overall: {'PASS' if results['all_passed'] else 'FAIL'}\n\n")
        for c in results["checks"]:
            fh.write(f"- **{c['name']}**: {'PASS' if c['passed'] else 'FAIL'}")
            fh.write(f" — {c.get('detail', c.get('error'))}\n")
        fh.write(f"\nJSON: `{out}`\n")

    print(f"\nWrote {md}")
    print(f"Wrote {out}")
    return 0 if results["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())