import numpy as np
import pandas as pd
import os
import json
import gc
import sys
from config import OMISSION_CONFIG, EPOCHS

def run_session_analysis(session_id, units_path):
    print(f"Part 1: Identifying top omission units for {session_id}...")
    metadata_path = f'D:/hnxj-gemini/ses-{session_id}_trials.csv'
    summary_path = f'D:/hnxj-gemini/ses-{session_id}_part1_summary.csv'
    
    if os.path.exists(summary_path):
        print(f"Summary for {session_id} already exists. Skipping.")
        return

    trials_df = pd.read_csv(metadata_path)
    spike_data = np.load(units_path, mmap_mode='r')
    n_units = spike_data.shape[0]
    time_axis = np.linspace(-1000, 4000, 6000)
    
    all_unit_data = []
    for u_idx in range(n_units):
        profile = np.mean(spike_data[u_idx, :, :], axis=0)
        
        # Analyze Mode 5 (Omission)
        m5_idx = np.where((time_axis >= 3000) & (time_axis <= 4500))[0]
        base_idx = np.where((time_axis >= -500) & (time_axis <= -100))[0]
        
        omit_firing = np.mean(profile[m5_idx])
        base_firing = np.mean(profile[base_idx])
        omit_index = omit_firing / (base_firing + 1e-6)
        
        all_unit_data.append({
            'unit_idx': u_idx,
            'omit_firing': float(omit_firing),
            'base_firing': float(base_firing),
            'omit_index': float(omit_index)
        })
        
    df = pd.DataFrame(all_unit_data)
    df.to_csv(summary_path, index=False)
    
    top_units = df.sort_values(by='omit_index', ascending=False).head(10)['unit_idx'].tolist()
    print(f"Top 10 Omission Units for {session_id}: {top_units}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python analyze_part1.py <session_id> <units_path>")
    else:
        run_session_analysis(sys.argv[1], sys.argv[2])
