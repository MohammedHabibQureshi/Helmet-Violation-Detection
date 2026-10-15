#!/usr/bin/env python3
"""Environment diagnostics for the Helmet Violation Detection project.

Records OS, Python, PyTorch/CUDA, GPU, and key package versions into both a
Markdown report and a machine-readable JSON file.
"""

import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"


def _safe_import(name):
    try:
        mod = __import__(name)
        return mod, getattr(mod, "__version__", None)
    except Exception as exc:  # pragma: no cover - environment dependent
        return None, f"IMPORT_ERROR: {exc}"


def _pip_version(pkg):
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pip", "show", pkg],
            capture_output=True, text=True, timeout=30,
        )
        for line in out.stdout.splitlines():
            if line.lower().startswith("version:"):
                return line.split(":", 1)[1].strip()
        return None
    except Exception:
        return None


def main() -> int:
    diag = {
        "generated_at": __import__("datetime").datetime.now().isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": sys.version,
            "python_executable": sys.executable,
        },
        "packages": {},
        "cuda": {},
        "gpu": {},
        "resources": {},
    }

    # Core packages
    for name in ("torch", "numpy", "cv2", "ultralytics", "easyocr",
                 "matplotlib", "pandas", "yaml", "PIL", "scipy"):
        mod, ver = _safe_import(name)
        diag["packages"][name] = {
            "version": ver,
            "available": mod is not None,
        }

    # CUDA / GPU
    try:
        import torch
        diag["packages"]["torch"]["build"] = getattr(torch.version, "cuda", None)
        diag["cuda"]["torch_available"] = torch.cuda.is_available()
        diag["cuda"]["device_count"] = torch.cuda.device_count()
        diag["cuda"]["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            diag["gpu"] = {
                "name": props.name,
                "major": props.major,
                "minor": props.minor,
                "total_memory_gb": round(props.total_memory / 1e9, 3),
                "device_count": torch.cuda.device_count(),
            }
            # real tensor op
            import time
            a = torch.randn(1024, 1024, device="cuda")
            b = torch.randn(1024, 1024, device="cuda")
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(20):
                c = a @ b
            torch.cuda.synchronize()
            diag["gpu"]["matmul_1024_ms"] = round(
                (time.perf_counter() - t0) / 20 * 1000.0, 3)
            diag["gpu"]["memory_allocated_gb"] = round(
                torch.cuda.memory_allocated() / 1e9, 3)
            del a, b, c
    except Exception as exc:
        diag["cuda"]["error"] = str(exc)

    # nvidia-smi
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20)
        diag["gpu"]["nvidia_smi"] = out.stdout.strip()
    except Exception as exc:
        diag["gpu"]["nvidia_smi_error"] = str(exc)

    # Resources
    try:
        import psutil
        diag["resources"]["ram_total_gb"] = round(
            psutil.virtual_memory().total / 1e9, 2)
        diag["resources"]["cpu_count"] = psutil.cpu_count(logical=True)
        diag["resources"]["cpu_count_physical"] = psutil.cpu_count(logical=False)
        diag["resources"]["disk_free_gb"] = round(
            shutil.disk_usage(str(PROJECT_ROOT)).free / 1e9, 2)
    except Exception as exc:
        diag["resources"]["error"] = str(exc)

    # Write JSON
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_DIR / "environment_diagnostics.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(diag, fh, indent=2, ensure_ascii=False, default=str)

    # Write Markdown
    md_path = REPORTS_DIR / "ENVIRONMENT_DIAGNOSTICS.md"
    with md_path.open("w", encoding="utf-8") as fh:
        fh.write("# Environment Diagnostics\n\n")
        fh.write(f"Generated: {diag['generated_at']}\n\n")
        fh.write("## Platform\n\n")
        for k, v in diag["platform"].items():
            fh.write(f"- **{k}**: `{v}`\n")
        fh.write("\n## Packages\n\n")
        fh.write("| Package | Version | Available |\n|---|---|---|\n")
        for name, info in diag["packages"].items():
            fh.write(f"| {name} | {info['version']} | {info['available']} |\n")
        fh.write("\n## CUDA / GPU\n\n")
        for k, v in diag["cuda"].items():
            fh.write(f"- **{k}**: `{v}`\n")
        fh.write("\n## GPU\n\n")
        for k, v in diag["gpu"].items():
            fh.write(f"- **{k}**: `{v}`\n")
        fh.write("\n## Resources\n\n")
        for k, v in diag["resources"].items():
            fh.write(f"- **{k}**: `{v}`\n")
        fh.write(f"\nJSON: `{json_path}`\n")

    print(f"Wrote {md_path}")
    print(f"Wrote {json_path}")
    print(f"CUDA available: {diag['cuda'].get('torch_available')}")
    print(f"GPU: {diag['gpu'].get('name')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())