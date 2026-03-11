import pynwb
from pynwb import NWBHDF5IO
import os
import json
import gc
import pandas as pd
import numpy as np

NWB_DIR = r'D:\CDOC\Analysis\reconstructed_nwbdata'
FILES = [f for f in os.listdir(NWB_DIR) if f.endswith('.nwb')]
OUTPUT_JSON = r'D:\OmissionAnalysis\nwb_quality_inspection.json'

def inspect_nwb_quality(file_path):
    print(f"--- Quality Inspection: {os.path.basename(file_path)} ---")
    try:
        with NWBHDF5IO(file_path, 'r', load_namespaces=True) as io:
            nwb = io.read()
            
            # 1. Probes & Area Metadata
            probes = {}
            if nwb.electrode_groups:
                for name, group in nwb.electrode_groups.items():
                    probes[name] = group.location
            
            # 2. Unit Quality Metrics
            # Target: presence_ratio > 0.9, SNR > 2.0 (or quality == 'good')
            n_total_units = 0
            n_good_units = 0
            unit_details = []
            
            if nwb.units:
                df_units = nwb.units.to_dataframe()
                n_total_units = len(df_units)
                
                # Filter for 'good' quality
                # Case 1: quality column exists (prioritize this)
                if 'quality' in df_units.columns:
                    mask_good = (df_units['quality'].astype(str).isin(['good', '1', '1.0', 'good_unit']))
                # Case 2: Use presence_ratio as proxy if quality is missing or for additional filtering
                else:
                    mask_good = (df_units.index == df_units.index) # All true
                    if 'presence_ratio' in df_units.columns:
                        mask_good &= (pd.to_numeric(df_units['presence_ratio'], errors='coerce') > 0.9)
                
                # Optional: Include SNR if quality is NOT 'good' but SNR is high (but we'll stick to quality first)
                n_good_units = int(mask_good.sum())
                
                # Breakdown of good units per probe
                if 'group_name' in df_units.columns:
                    unit_details = df_units[mask_good]['group_name'].value_counts().to_dict()
            
            # 3. Trial Completeness (Target: 960 correct full omission trials)
            intervals = nwb.intervals
            target_table = intervals.get('omission_glo_passive')
            
            n_correct_total = 0
            n_960_target_reached = False
            mode_counts = {}
            
            if target_table:
                df_trials = target_table.to_dataframe()
                # Filter correct trials (string-aware)
                mask_correct = (df_trials['correct'].astype(str) == '1.0')
                correct_trials = df_trials[mask_correct].drop_duplicates(subset=['trial_num'])
                
                n_correct_total = len(correct_trials)
                n_960_target_reached = (n_correct_total >= 960)
                
                # Breakdown of key omission modes
                if 'task_condition_number' in correct_trials.columns:
                    counts = correct_trials['task_condition_number'].value_counts()
                    mode_counts = {str(int(float(k))): int(v) for k, v in counts.items()}
            
            res = {
                "file": os.path.basename(file_path),
                "probes": probes,
                "units": {
                    "total": n_total_units,
                    "good_quality": n_good_units,
                    "good_per_probe": unit_details
                },
                "trials": {
                    "total_correct": n_correct_total,
                    "is_960_complete": n_960_target_reached,
                    "modes": mode_counts
                },
                "status": "success"
            }
            return res
    except Exception as e:
        print(f"!!! Error: {e}")
        return {"file": os.path.basename(file_path), "error": str(e), "status": "failed"}

if __name__ == "__main__":
    results = []
    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON, "r") as f:
            try: results = json.load(f)
            except: results = []
    
    processed = [r['file'] for r in results]
    
    for f in FILES:
        if f in processed: continue
        data = inspect_nwb_quality(os.path.join(NWB_DIR, f))
        results.append(data)
        with open(OUTPUT_JSON, "w") as jf:
            json.dump(results, jf, indent=4)
        gc.collect()
    
    print(f"Quality inspection complete. Results in {OUTPUT_JSON}")
