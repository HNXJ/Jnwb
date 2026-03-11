import pynwb
from pynwb import NWBHDF5IO
import os
import json
import gc
import pandas as pd

NWB_DIR = r'D:\CDOC\Analysis\reconstructed_nwbdata'
FILES = [f for f in os.listdir(NWB_DIR) if f.endswith('.nwb')]
OUTPUT_JSON = r'D:\OmissionAnalysis\nwb_inspection_results.json'

def inspect_nwb(file_path):
    print(f"--- Opening {os.path.basename(file_path)} ---")
    try:
        with NWBHDF5IO(file_path, 'r', load_namespaces=True) as io:
            nwb = io.read()
            
            # 1. Probes & Areas
            areas = []
            if nwb.electrode_groups:
                for name, group in nwb.electrode_groups.items():
                    areas.append(f"{name}: {group.location}")
            
            # 2. Neurons (Units)
            n_units = len(nwb.units) if nwb.units else 0
            
            # 3. Trials per Condition
            intervals = nwb.intervals
            trial_stats = {}
            target_table = intervals.get('omission_glo_passive') or intervals.get('trials')
            
            if target_table:
                df = target_table.to_dataframe()
                if 'task_condition_number' in df.columns:
                    # Filter for correct trials, handling potential string/float differences
                    if 'correct' in df.columns:
                        mask = (df['correct'].astype(str) == '1.0')
                    else:
                        mask = (df.index == df.index) # All trials
                    
                    # Group by unique trial_num to avoid counting event rows as trials
                    if 'trial_num' in df.columns:
                        unique_trials = df[mask].drop_duplicates(subset=['trial_num'])
                        counts = unique_trials['task_condition_number'].value_counts()
                    else:
                        counts = df[mask]['task_condition_number'].value_counts()
                    
                    trial_stats = {str(int(float(k))) if not pd.isna(k) else "NaN": int(v) for k, v in counts.items()}
            
            res = {
                "file": os.path.basename(file_path),
                "areas": areas,
                "n_units": n_units,
                "trial_breakdown": trial_stats,
                "status": "success"
            }
            return res
    except Exception as e:
        print(f"!!! Error inspecting {file_path}: {e}")
        return {"file": os.path.basename(file_path), "error": str(e), "status": "failed"}

if __name__ == "__main__":
    results = []
    
    # Load existing if available to resume
    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON, "r") as f:
            try:
                results = json.load(f)
            except:
                results = []
    
    processed_files = [r['file'] for r in results]
    
    for f in FILES:
        if f in processed_files:
            print(f"Skipping {f}, already processed.")
            continue
            
        data = inspect_nwb(os.path.join(NWB_DIR, f))
        results.append(data)
        
        # Immediate Save
        with open(OUTPUT_JSON, "w") as jf:
            json.dump(results, jf, indent=4)
            
        # Hard Clear
        gc.collect()
        
    print(f"Inspection process finished. Results in {OUTPUT_JSON}")
