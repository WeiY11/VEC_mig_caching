import sys
import os
print(f"CWD: {os.getcwd()}")
try:
    import Benchmarks.nath_dynamic_offload_heuristic
    print("Old file imported!")
except ImportError:
    print("Old file NOT imported (Good)")

try:
    import Benchmarks.nath_dynamic_offload_heuristic_v2 as m
    print(f"New file imported: {m.__file__}")
    from Benchmarks.nath_dynamic_offload_heuristic_v2 import DynamicOffloadHeuristic
    import inspect
    print(f"Class source match 'DEBUG': {'DEBUG' in inspect.getsource(DynamicOffloadHeuristic._parse_state)}")
except ImportError as e:
    print(f"New file NOT imported: {e}")
