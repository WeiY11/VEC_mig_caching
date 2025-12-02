import os
import glob

results_dir = r"d:\VEC_mig_caching\results\single_agent\optimized_td3"
files = glob.glob(os.path.join(results_dir, "*"))
files.sort(key=os.path.getmtime)

if files:
    print(f"Latest file: {files[-1]}")
    print(f"All files: {[os.path.basename(f) for f in files]}")
else:
    print("No files found.")
