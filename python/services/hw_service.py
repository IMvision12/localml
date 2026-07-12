"""Hardware sampling - Python port of `services/systeminfo.js`.

Returns the same shape the renderer consumes: cpu / mem / gpu / disk / os.
psutil for CPU/mem/disk; nvidia-smi (falling back to torch) for GPU.
"""
from __future__ import annotations

import platform
import shutil
import subprocess

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None


def _probe_nvidia():
    if platform.system() == "Darwin" or not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2.0,
        )
        line = (out.stdout or "").strip().splitlines()
        if not line:
            return None
        parts = [p.strip() for p in line[0].split(",")]
        name, mem_used, mem_total, util, temp = parts[:5]
        return {
            "model": name,
            "memUsed": int(float(mem_used)) * 1024 * 1024,
            "memTotal": int(float(mem_total)) * 1024 * 1024,
            "utilization": int(float(util)),
            "temperature": int(float(temp)),
        }
    except Exception:
        return None


def _torch_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            free, total = torch.cuda.mem_get_info(0)
            return {
                "model": props.name, "vendor": "NVIDIA", "unified": False,
                "vram": total, "memTotal": total, "memUsed": total - free,
                "utilization": None, "source": "torch.cuda",
            }
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is not None and mps.is_available():
            vm = psutil.virtual_memory() if psutil else None
            return {
                "model": "Apple Silicon GPU", "vendor": "Apple", "unified": True,
                "vram": vm.total if vm else 0, "memTotal": vm.total if vm else 0,
                "memUsed": (vm.total - vm.available) if vm else 0,
                "utilization": None, "source": "torch.mps",
            }
    except Exception:
        pass
    return None


def sample_hw() -> dict:
    try:
        if psutil is None:
            return {"error": "psutil not installed"}

        vm = psutil.virtual_memory()
        freq = None
        try:
            freq = psutil.cpu_freq()
        except Exception:
            freq = None
        load = psutil.cpu_percent(interval=None)

        try:
            du = psutil.disk_usage("/" if platform.system() != "Windows" else "C:\\")
        except Exception:
            du = None

        nv = _probe_nvidia()
        if nv:
            gpu = {
                "model": nv["model"], "vendor": "NVIDIA", "unified": False,
                "vram": nv["memTotal"], "memTotal": nv["memTotal"], "memUsed": nv["memUsed"],
                "utilization": nv["utilization"], "temperature": nv.get("temperature"),
                "source": "nvidia-smi",
            }
        else:
            gpu = _torch_gpu() or {
                "model": "Integrated GPU", "vendor": "", "unified": False,
                "vram": 0, "memTotal": 0, "memUsed": 0, "utilization": None,
                "source": "none",
            }

        uname = platform.uname()
        return {
            "cpu": {
                "brand": (uname.processor or platform.processor() or "CPU"),
                "cores": psutil.cpu_count(logical=False) or psutil.cpu_count() or 0,
                "threads": psutil.cpu_count(logical=True) or 0,
                "speed": round((freq.max or freq.current) / 1000, 2) if freq else 0,
                "load": round(load),
            },
            "mem": {
                "total": vm.total, "used": vm.total - vm.available,
                "free": vm.available, "pct": round(vm.percent),
            },
            "gpu": gpu,
            "disk": {
                "total": du.total if du else 0,
                "used": du.used if du else 0,
                "free": du.free if du else 0,
                "mount": "C:\\" if platform.system() == "Windows" else "/",
            },
            "os": {
                "platform": platform.system().lower(),
                "distro": platform.system(),
                "release": platform.release(),
                "build": platform.version(),
                "arch": platform.machine(),
            },
        }
    except Exception as e:
        return {"error": str(e)}
