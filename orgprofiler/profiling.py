import time
import contextlib
import os
import sys
import platform
from typing import Dict, Optional, Any
from loguru import logger

try:
    import psutil
except ImportError:
    psutil = None

def get_max_resident_set_size_bytes() -> Optional[int]:
    """Get the maximum resident set size (peak memory usage) in bytes."""
    try:
        import resource
        max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform.startswith("linux"):
            return int(max_rss) * 1024
        return int(max_rss)
    except Exception:
        return None

@contextlib.contextmanager
def time_block(label: str):
    _t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - _t0
        logger.info(f"[TIMER] {label}: {dt:.3f}s")

@contextlib.contextmanager
def timed(timings: Dict[str, float], key: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        timings[key] = round(time.perf_counter() - t0, 6)

class ResourceProfiler:
    def __init__(self, label: str = "analyze"):
        self.label = label
        self.metrics: Dict[str, Any] = {}

    def __enter__(self):
        self.t0 = time.perf_counter()
        if psutil:
            self.proc = psutil.Process(os.getpid())
            self.ct0 = self.proc.cpu_times()
            self.mem0 = self.proc.memory_info().rss
        else:
            self.proc = None
            self.ct0 = None
            self.mem0 = None
        self.maxrss0 = get_max_resident_set_size_bytes()
        return self

    def __exit__(self, exc_type, exc, tb):
        t1 = time.perf_counter()
        wall_s = t1 - self.t0

        cpu_user_s = cpu_sys_s = rss_now = rss_delta = None
        if self.proc:
            ct1 = self.proc.cpu_times()
            mi1 = self.proc.memory_info()
            cpu_user_s = (ct1.user - self.ct0.user)
            cpu_sys_s  = (ct1.system - self.ct0.system)
            rss_now    = mi1.rss
            rss_delta  = (rss_now - self.mem0)

        maxrss1 = get_max_resident_set_size_bytes()
        peak_rss_bytes = None
        if maxrss1 is not None and self.maxrss0 is not None:
            peak_rss_bytes = max(0, maxrss1 - self.maxrss0) or maxrss1

        def bytes_to_mb(bytes_value):
            return None if bytes_value is None else round(bytes_value / (1024*1024), 3)

        self.metrics = {
            "wall_time_s": round(wall_s, 6),
            "cpu_user_s": None if cpu_user_s is None else round(cpu_user_s, 6),
            "cpu_sys_s": None if cpu_sys_s is None else round(cpu_sys_s, 6),
            "rss_now_bytes": rss_now,
            "rss_now_mb": bytes_to_mb(rss_now),
            "rss_delta_bytes": rss_delta,
            "rss_delta_mb": bytes_to_mb(rss_delta),
            "peak_rss_bytes": peak_rss_bytes,
            "peak_rss_mb": bytes_to_mb(peak_rss_bytes),
            "platform": platform.platform(),
        }

        logger.info(
            f"[{self.label}] wall={wall_s:.3f}s "
            f"cpu_user={self.metrics['cpu_user_s']}s cpu_sys={self.metrics['cpu_sys_s']}s "
            f"rss_now={self.metrics['rss_now_mb']}MB Δrss={self.metrics['rss_delta_mb']}MB "
            f"peak_rss={self.metrics['peak_rss_mb']}MB"
        )

