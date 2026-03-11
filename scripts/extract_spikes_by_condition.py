"""
Extracts spiking activity for good units, aligned to fixation onset,
and organized by the 12 OGLO trial conditions.
"""
import sys
import os
import h5py
import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO
import gc
import json

# Add the jnwb package to the Python path
sys.path.append(r'D:\jnwb')
from jnwb.oglo import extract_good_units
from jnwb.oglo_v2 import get_oglo_trial_masks_v2 as get_trial_masks

def extract_spikes_for_session(nwb_path: str, output_path: str):
    """
    Processes a single NWB file to extract and save spiking data by condition.
    """
    print(f"--- Starting Spike Extraction for {os.path.basename(nwb_path)} ---")

    # --- 1. Load Data & Metadata ---
    with NWBHDF5IO(nwb_path, 'r', load_namespaces=True) as io:
        nwb = io.read()
        
        # Get trial information
        trials_df = nwb.intervals['omission_glo_passive'].to_dataframe()
        
        # Get 'good quality' units and their original indices in the NWB file
        good_units_df = extract_good_units(nwb)
        good_unit_indices = good_units_df.index.tolist()
        
        print(f"Found {len(good_units_df)} good quality units out of {len(nwb.units)} total.")

        # --- 2. Identify & Filter Trials ---
        # Filter for correct trials only
        correct_trials_df = trials_df[trials_df['correct'].astype(str) == '1.0'].drop_duplicates(subset=['trial_num']).sort_values('start_time')
        print(f"Found {len(correct_trials_df)} correct trials.")

        # Get the condition masks for these correct trials
        trial_masks = get_trial_masks(correct_trials_df)

        # --- 3. Prepare Output Data Structure ---
        # We will store data in a dictionary first, then save to HDF5
        # { 'AAAB': {'spikes': np.ndarray, 'info': pd.DataFrame}, ... }
        results = {}

        # --- 4. Extract Spiking Data for each Condition ---
        fs = 1000 # Spiking data will be binned at 1kHz (1ms bins)
        pre_s, post_s = 1.0, 5.0
        n_samples = int((pre_s + post_s) * fs)
        
        for condition_name, mask in trial_masks.items():
            condition_trials = correct_trials_df[mask]
            n_cond_trials = len(condition_trials)
            
            if n_cond_trials == 0:
                continue

            print(f"  Processing '{condition_name}': {n_cond_trials} trials...")
            
            # Data array for this condition: (trials, units, time)
            spikes_array = np.zeros((n_cond_trials, len(good_units_df), n_samples), dtype=np.uint8)
            
            # Info array for this condition
            trial_info = []

            for i, (trial_idx, trial_row) in enumerate(condition_trials.iterrows()):
                fixation_start = trial_row['start_time']
                window_start = fixation_start - pre_s
                window_end = fixation_start + post_s
                
                trial_info.append({
                    'trial_num': trial_row['trial_num'],
                    'condition': condition_name,
                    'start_time': fixation_start
                })

                # Extract spikes for all good units for this trial
                for j, unit_nwb_idx in enumerate(good_unit_indices):
                    spike_times = nwb.units.get_unit_spike_times(unit_nwb_idx)
                    
                    # Filter spikes within the 6s window
                    trial_spike_times = spike_times[(spike_times >= window_start) & (spike_times < window_end)]
                    
                    # Bin spikes into 1ms bins (relative to window start)
                    relative_spike_times = trial_spike_times - window_start
                    bin_indices = (relative_spike_times * fs).astype(int)
                    
                    # Ensure indices are within bounds
                    bin_indices = bin_indices[bin_indices < n_samples]
                    
                    spikes_array[i, j, bin_indices] = 1 # Mark spike presence in bin
            
            results[condition_name] = {
                'spikes': spikes_array,
                'info': pd.DataFrame(trial_info)
            }
            
            gc.collect()

    # --- 5. Save to HDF5 ---
    print(f"Saving extracted data to {output_path}...")
    with h5py.File(output_path, 'w') as f:
        for condition, data in results.items():
            grp = f.create_group(condition)
            grp.create_dataset('spiking_activity', data=data['spikes'], compression='gzip')
            # Save trial info
            info_df = data['info']
            # Manually create a structured numpy array
            structured_array = np.zeros(len(info_df), dtype=[('trial_num', 'i4'), ('condition', 'S10'), ('start_time', 'f8')])
            structured_array['trial_num'] = info_df['trial_num'].astype(float).astype(int).values
            structured_array['condition'] = info_df['condition'].str.encode('utf-8')
            structured_array['start_time'] = info_df['start_time'].values
            grp.create_dataset('trial_info', data=structured_array)
            
    print("--- Extraction Complete ---")
    return {
        'total_good_units': len(good_units_df),
        'total_correct_trials': len(correct_trials_df),
        'conditions_found': list(results.keys())
    }


if __name__ == "__main__":
    SESSION_ID = "230818" # Test session
    NWB_FILE = f'D:/CDOC/Analysis/reconstructed_nwbdata/sub-C31o_ses-{SESSION_ID}_rec.nwb'
    OUTPUT_FILE = f'D:/OmissionAnalysis/spikes_by_condition_ses-{SESSION_ID}.h5'
    
    report = extract_spikes_for_session(NWB_FILE, OUTPUT_FILE)
    print("\n--- REPORT ---")
    print(json.dumps(report, indent=4))
