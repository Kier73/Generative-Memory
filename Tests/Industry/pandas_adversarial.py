import os
import sys
import psutil
import pandas as pd
import numpy as np
import time

# Ensure gmem is in your PYTHONPATH or run from the project root
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gmem import GMemContext

def get_rss_mib():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

def adversarial_pandas_test():
    print("=== Adversarial Pandas Stress Test (Target < 200MB RAM) ===")
    initial_mem = get_rss_mib()
    print(f"Initial RSS: {initial_mem:.2f} MiB")

    # 1. Probe the entire 16-EB space at log-scale
    print("\n[Phase 1] Probing 16-EB space at log-scale (2^0 to 2^63)...")
    ctx1 = GMemContext(seed=0xAAAA)
    ctx2 = GMemContext(seed=0xBBBB)
    
    addresses = [2**i for i in range(64)] + [2**64 - 1]
    
    df1 = pd.DataFrame({
        'addr': addresses,
        'val_a': [ctx1[a] for a in addresses]
    })
    
    df2 = pd.DataFrame({
        'addr': addresses,
        'val_b': [ctx2[a] for a in addresses]
    })
    
    print(f"RSS after creating probe DataFrames: {get_rss_mib():.2f} MiB")

    # 2. Relational Merge at extreme offsets
    print("\n[Phase 2] Performing join/merge between two 16-EB manifolds...")
    merged = pd.merge(df1, df2, on='addr')
    merged['diff'] = merged['val_a'] - merged['val_b']
    
    print(f"Merged Result Head:\n{merged.head()}")
    print(f"RSS after merge: {get_rss_mib():.2f} MiB")

    # 3. Virtual Windowing (sampling slices across EB space)
    print("\n[Phase 3] Slicing the Infinite: Windowed stats across TB, PB, and EB offsets...")
    windows = {
        'TB': 10**12,
        'PB': 10**15,
        'EB': 10**18
    }
    
    window_size = 10000
    all_data = []
    
    for label, offset in windows.items():
        print(f"  Sampling {window_size} values at {label} offset...")
        vals = ctx1.fetch_bulk(offset, window_size)
        tmp_df = pd.DataFrame({'val': vals, 'offset_type': label})
        all_data.append(tmp_df)
        print(f"  Current RSS: {get_rss_mib():.2f} MiB")

    full_manifold_df = pd.concat(all_data, ignore_index=True)
    
    stats = full_manifold_df.groupby('offset_type')['val'].agg(['mean', 'std', 'skew'])
    print(f"\nStatistical consistency check across offsets:\n{stats}")
    
    print(f"\n[Phase 4] Massive search & match across indices...")
    # Find index nearest to 0.777 in the EB space
    target = 0.777
    idx = ctx1.search(target)
    found_val = ctx1.fetch_monotonic(idx)
    print(f"Found target {target} at manifold index {idx} (value: {found_val:.6f})")

    final_mem = get_rss_mib()
    print(f"\nFinal RSS: {final_mem:.2f} MiB")
    print(f"Total RSS Delta: {final_mem - initial_mem:.2f} MiB")
    
    if final_mem < 200:
        print("SUCCESS: Memory remained under 200 MiB limit.")
    else:
        print("FAILURE: Memory limit exceeded.")

if __name__ == "__main__":
    adversarial_pandas_test()
