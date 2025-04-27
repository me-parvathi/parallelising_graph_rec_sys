# profiling.py

import cProfile
import pstats
import io
from memory_profiler import memory_usage
from functools import wraps
import time

# NOT USED: This functionality is now covered by the main profiling system in main.py
"""
def profile_func(func):
    """
    Decorator to:
      - run cProfile and print top 10 cumulative calls
      - measure simple CPU memory delta via memory_profiler
      - measure wall-clock time
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # start profiling
        pr = cProfile.Profile()
        pr.enable()

        # capture starting memory (list of floats, MiB)
        mem_start = memory_usage(-1, interval=0.1, timeout=1)
        t0 = time.time()

        result = func(*args, **kwargs)

        # stop profiling
        pr.disable()
        elapsed = time.time() - t0
        mem_end = memory_usage(-1, interval=0.1, timeout=1)

        # dump top‐10 by cumulative time
        s = io.StringIO()
        stats = pstats.Stats(pr, stream=s).sort_stats("cumulative")
        stats.print_stats(10)
        print("\n=== cProfile top 10 cumulative ===\n", s.getvalue())

        # simple memory delta
        delta_mem = max(mem_end) - min(mem_start)
        print(f"[PROFILE] {func.__name__} took {elapsed:.3f}s, "
              f"≈{delta_mem:.2f} MiB RAM increase\n")

        return result

    return wrapper
"""
