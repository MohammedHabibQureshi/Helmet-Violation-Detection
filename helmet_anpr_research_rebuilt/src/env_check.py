"""Environment checker for the Helmet Violation Detection project.

Collects hardware/software information and saves it to
results/environment.json.
"""

import json
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src import PROJECT_ROOT, RESULTS_DIR

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cv2

    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try:
    from ultralytics import version as ultralytics_version
except ImportError:
    ultralytics_version = None


def _get_nvidia_smi() -> dict:
    """Parse nvidia-smi output for GPU info (single GPU)."""
    result = {
        "gpu_name": None, "gpu_driver_version": None, "gpu_vram_gb": None
    }
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,"
             "driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            parts = [p.strip() for p in proc.stdout.strip().split(",") if p.strip()]
            if len(parts) >= 3:
                result["gpu_name"] = parts[0]
                try:
                    result["gpu_vram_gb"] = round(float(parts[1]) / 1024, 2)
                except ValueError:
                    pass
                result["gpu_driver_version"] = parts[3]
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return result


def _parse_nvml() -> dict:
    """Parse GPU info via pynvml."""
    result = {
        "gpu_name": None, "gpu_driver_version": None, "gpu_vram_gb": None
    }
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        result["gpu_name"] = pynvml.nvmlDeviceGetName(handle).decode("utf-8")
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        result["gpu_vram_gb"] = round(info.total / (1024 ** 3), 2)
        pynvml.nvmlShutdown()
    except ImportError:
        pass
    except Exception:
        pass
    return result


def _get_cpu_info() -> dict:
    """Get CPU name, cores, and frequency."""
    result = {
        "cpu_name": None,
        "cpu_cores_physical": None,
        "cpu_cores_logical": None,
        "cpu_freq_mhz": None,
    }
    # CPU name
    try:
        cpu_name = platform.processor()
        result["cpu_name"] = cpu_name if cpu_name else platform.machine()
    except Exception:
        result["cpu_name"] = "unknown"
    # CPU cores via lscpu
    try:
        proc = subprocess.run(
            ["lscpu"], capture_output=True, text=True, timeout=15, check=False,
        )
        if proc.returncode == 0:
            for line in proc.stdout.splitlines():
                if line.startswith("Model name:"):
                    result["cpu_name"] = line.split(":", 1)[1].strip()
                elif line.startswith("Physical cores:"):
                    try:
                        result["cpu_cores_physical"] = int(
                            line.split(":", 1)[1].strip())
                    except ValueError:
                        pass
                elif line.startswith("Virtual cores:"):
                    try:
                        result["cpu_cores_logical"] = int(
                            line.split(":", 1)[1].strip())
                    except ValueError:
                        pass
                elif line.startswith("CPU MHz:"):
                    try:
                        result["cpu_freq_mhz"] = round(
                            float(line.split(":", 1)[1].strip()), 2
                        )
                    except ValueError:
                        pass
    except Exception:
        pass
    # Fallback: multiprocessing cpu_count
    try:
        from multiprocessing import cpu_count
        if result["cpu_cores_logical"] is None:
            result["cpu_cores_logical"] = cpu_count()
    except Exception:
        pass
    # Windows fallback for cores/frequency
    if result["cpu_cores_logical"] is None or result["cpu_freq_mhz"] is None:
        try:
            proc = subprocess.run(
                ["powershell", "-Command",
                 "Get-CimInstance Win32_Processor | "
                 "Select-Object Name,NumberOfCores,NumberOfLogicalProcessors,"
                 "MaxClockSpeed | ConvertTo-Json"],
                capture_output=True, text=True, timeout=15, check=False,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                import json as _json
                for obj in _json.loads(proc.stdout):
                    if isinstance(obj, dict):
                        if result["cpu_name"] == "unknown":
                            result["cpu_name"] = obj.get("Name")
                        if result["cpu_cores_logical"] is None:
                            result["cpu_cores_logical"] = \
                                obj.get("NumberOfLogicalProcessors")
                        if result["cpu_cores_physical"] is None:
                            result["cpu_cores_physical"] = \
                                obj.get("NumberOfCores")
                        if result["cpu_freq_mhz"] is None and \
                                obj.get("MaxClockSpeed"):
                            result["cpu_freq_mhz"] = int(obj["MaxClockSpeed"])
        except Exception:
            pass
    if result["cpu_cores_logical"] is None:
        result["cpu_cores_logical"] = "unknown"
    if result["cpu_cores_physical"] is None:
        result["cpu_cores_physical"] = result["cpu_cores_logical"]
    return result


def _get_memory_info() -> dict:
    """Get RAM total and available in GB."""
    result = {"ram_total_gb": None, "ram_available_gb": None}
    try:
        from psutil import virtual_memory

        mem = virtual_memory()
        result["ram_total_gb"] = round(mem.total / (1024 ** 3), 2)
        result["ram_available_gb"] = round(mem.available / (1024 ** 3), 2)
    except ImportError:
        try:
            proc = subprocess.run(
                ["powershell", "-Command",
                 "Get-CimInstance Win32_OperatingSystem | "
                 "Select-Object TotalVisibleMemorySize,FreePhysicalMemory | "
                 "ConvertTo-Json"],
                capture_output=True, text=True, timeout=15, check=False,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                import json as _json
                obj = _json.loads(proc.stdout)
                if isinstance(obj, dict):
                    result["ram_total_gb"] = round(
                        obj.get("TotalVisibleMemorySize", 0) / (1024 * 1024), 2
                    )
                    result["ram_available_gb"] = round(
                        obj.get("FreePhysicalMemory", 0) / (1024 * 1024), 2
                    )
        except Exception:
            pass
    return result


def _get_disk_info(root: Path) -> dict:
    """Get free disk space at root in GB."""
    result = {"disk_free_gb": None}
    try:
        from psutil import disk_usage

        usage = disk_usage(str(root))
        result["disk_free_gb"] = round(usage.free / (1024 ** 3), 2)
    except ImportError:
        try:
            statvfs = __import__("os").statvfs(str(root))
            result["disk_free_gb"] = round(
                statvfs.f_bavail * statvfs.f_frsize / (1024 ** 3), 2
            )
        except Exception:
            pass
    return result


def _get_tesseract_installed() -> bool:
    """Check if tesseract CLI is available."""
    try:
        result = subprocess.run(
            ["tesseract", "--version"], capture_output=True, text=True,
            timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def collect_environment(project_root: Path = None) -> dict:
    """Collect comprehensive environment information."""
    root = project_root or PROJECT_ROOT
    results = {}

    # Python / platform
    results["python_version"] = ".".join(
        str(v) for v in sys.version_info[:3]
    )
    results["platform"] = platform.system()
    results["platform_release"] = platform.release()
    results["platform_version"] = platform.version()
    results["hostname"] = socket.gethostname()
    results["os_version"] = platform.version()

    # Libraries
    results["torch_version"] = torch.__version__ if TORCH_AVAILABLE else None
    results["torch_available"] = TORCH_AVAILABLE
    results["cuda_available"] = (
        torch.cuda.is_available() if TORCH_AVAILABLE else False
    )
    if TORCH_AVAILABLE and torch.cuda.is_available():
        results["cuda_version"] = torch.version.cuda
        results["cudnn_version"] = torch.backends.cudnn.version()
    else:
        results["cuda_version"] = None
        results["cudnn_version"] = None
    results["opencv_version"] = cv2.__version__ if OPENCV_AVAILABLE else None
    results["opencv_available"] = OPENCV_AVAILABLE
    results["ultralytics_version"] = ultralytics_version or None

    # OCR engines
    try:
        import easyocr

        results["easyocr_installed"] = True
        results["easyocr_version"] = easyocr.__version__
    except ImportError:
        results["easyocr_installed"] = False
        results["easyocr_version"] = None
    try:
        import paddleocr

        results["paddleocr_installed"] = True
        results["paddleocr_version"] = paddleocr.__version__
    except ImportError:
        results["paddleocr_installed"] = False
        results["paddleocr_version"] = None
    try:
        import pytesseract

        results["pytesseract_installed"] = True
    except ImportError:
        results["pytesseract_installed"] = False
    results["tesseract_installed"] = _get_tesseract_installed()

    # CPU
    results["cpu_info"] = _get_cpu_info()

    # Memory
    results["memory_info"] = _get_memory_info()

    # GPU
    gpu_info = _get_nvidia_smi()
    if not gpu_info.get("gpu_name"):
        gpu_info = _parse_nvml()
    results["gpu_info"] = gpu_info

    # Disk
    results["disk_info"] = _get_disk_info(root)

    # Misc
    results["git_sha"] = None
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=root,
            capture_output=True, text=True, timeout=10, check=False,
        )
        if sha.returncode == 0:
            results["git_sha"] = sha.stdout.strip()
    except Exception:
        pass
    try:
        config_path = root / "configs" / "base_config.yaml"
        if config_path.exists():
            cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            results["project_root"] = str(cfg.get("project_root"))
            results["video_path"] = str(cfg.get("video_path"))
    except Exception:
        pass
    results["collection_timestamp"] = datetime.now(timezone.utc).isoformat()
    results["collection_timestamp_local"] = datetime.now().isoformat()

    return results


def save_environment_json(
    environment: dict, output_path: Path = None
) -> Path:
    """Save environment dict to a JSON file."""
    out = output_path or RESULTS_DIR / "environment.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(environment, fh, indent=2, ensure_ascii=False, default=str)
    return out


def load_environment_json(path: Path = None) -> dict:
    """Load environment JSON."""
    p = path or RESULTS_DIR / "environment.json"
    if not p.exists():
        raise FileNotFoundError(f"Environment JSON not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    """Entry point."""
    print("=" * 60)
    print("Helmet Violation Detection - Environment Check")
    print("=" * 60)
    try:
        environment = collect_environment()
        env_path = save_environment_json(environment)
        print(f"\nEnvironment saved to: {env_path}")
        print(f"\nPython:        {environment['python_version']}")
        print(f"Platform:      {environment['platform']} "
              f"{environment['platform_release']}")
        print(f"OS Version:    {environment['os_version']}")
        print(f"Torch:         {environment['torch_version']}")
        print(f"CUDA:          {environment['cuda_available']} "
              f"({'v' + str(environment['cuda_version'])})")
        print(f"OpenCV:        {environment['opencv_version']}")
        print(f"Ultralytics:   {environment['ultralytics_version']}")
        print(f"EasyOCR:       {environment['easyocr_installed']}")
        print(f"PaddleOCR:     {environment['paddleocr_installed']}")
        print(f"Pytesseract:   {environment['pytesseract_installed']}")
        print(f"Tesseract CLI: {environment['tesseract_installed']}")
        cpu = environment.get("cpu_info", {})
        print(f"CPU:           {cpu.get('cpu_name')} "
              f"({cpu.get('cpu_cores_physical')}C/"
              f"{cpu.get('cpu_cores_logical')}t)")
        mem = environment.get("memory_info", {})
        print(f"RAM:           {mem.get('ram_total_gb')} GB total, "
              f"{mem.get('ram_available_gb')} GB available")
        gpu = environment.get("gpu_info", {})
        print(f"GPU:           {gpu.get('gpu_name')} "
              f"({gpu.get('gpu_vram_gb')} GB VRAM)")
        disk = environment.get("disk_info", {})
        print(f"Disk Free:     {disk.get('disk_free_gb')} GB")
        return 0
    except Exception as exc:
        print(f"\nERROR: failed to collect environment: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
