# metrics.py

import time
import json
import psutil
import torch

# NOT USED: This class is kept for reference but not used in the codebase
"""
class Benchmark:
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.epochs = []

    def start_epoch(self):
        if not self.enabled:
            return
        # reset GPU‐peak counter
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._t0 = time.time()
        self._mem0 = psutil.Process().memory_info().rss / (1024 ** 2)

    def end_epoch(self, epoch_idx):
        if not self.enabled:
            return
        elapsed = time.time() - self._t0
        mem_now = psutil.Process().memory_info().rss / (1024 ** 2)
        gpu_peak = (torch.cuda.max_memory_allocated() / (1024 ** 2)
                    if torch.cuda.is_available() else None)
        self.epochs.append({
            "epoch": epoch_idx,
            "time_s": round(elapsed, 3),
            "cpu_mem_mib": round(mem_now - self._mem0, 2),
            "gpu_peak_mem_mib": round(gpu_peak, 2) if gpu_peak is not None else None,
        })

    def report(self):
        print("\n=== BENCHMARK REPORT ===")
        print(json.dumps({"epochs": self.epochs}, indent=2))
"""
