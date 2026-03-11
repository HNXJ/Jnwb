"""
Batch extracts spiking activity for all NWB sessions into separate HDF5 files.
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
    print(f"--- Starting Spike Extraction for {os.path.basename(nwb_path)} ---")

    if os.path.exists(output_path):
        print("Output file already exists. Skipping.")
        return
        
    with NWBHDF5IO(nwb_path, 'r', load_namespaces=True) as io:
        nwb = io.read()
        trials_df = nwb.intervals['omission_glo_passive'].to_dataframe()
        good_units_df = extract_good_units(nwb)
        good_unit_indices = good_units_df.index.tolist()
        
        print(f"Found {len(good_units_df)} good units.")

        correct_trials_df = trials_df[trials_df['correct'].astype(str) == '1.0'].drop_duplicates(subset=['trial_num'])
        trial_masks = get_trial_masks(correct_trials_df)
        
        results = {}
        fs = 1000
        pre_s, post_s = 1.0, 5.0
        n_samples = int((pre_s + post_s) * fs)

        for condition_name, mask in trial_masks.items():
            condition_trials = correct_trials_df[mask]
            if len(condition_trials) == 0: continue
            
            print(f"  Processing '{condition_name}': {len(condition_trials)} trials...")
            
            spikes_array = np.zeros((len(condition_trials), len(good_units_df), n_samples), dtype=np.uint8)
            trial_info = []

            for i, (_, row) in enumerate(condition_trials.iterrows()):
                window_start = row['start_time'] - pre_s
                window_end = row['start_time'] + post_s
                
                trial_info.append({'trial_num': row['trial_num'], 'condition': condition_name, 'start_time': row['start_time']})
                
                for j, unit_idx in enumerate(good_unit_indices):
                    spike_times = nwb.units.get_unit_spike_times(unit_idx)
                    trial_spikes = spike_times[(spike_times >= window_start) & (spike_times < window_end)]
                    bin_indices = ((trial_spikes - window_start) * fs).astype(int)
                    spikes_array[i, j, bin_indices[bin_indices < n_samples]] = 1
            
            results[condition_name] = {'spikes': spikes_array, 'info': pd.DataFrame(trial_info)}
            gc.collect()

    with h5py.File(output_path, 'w') as f:
        for condition, data in results.items():
            grp = f.create_group(condition)
            grp.create_dataset('spiking_activity', data=data['spikes'], compression='gzip')
            info_df = data['info']
            structured_array = np.zeros(len(info_df), dtype=[('trial_num', 'i4'), ('condition', 'S10'), ('start_time', 'f8')])
            structured_array['trial_num'] = info_df['trial_num'].astype(float).astype(int).values
            structured_array['condition'] = info_df['condition'].str.encode('utf-8')
            structured_array['start_time'] = info_df['start_time'].values
            grp.create_dataset('trial_info', data=structured_array)

if __name__ == "__main__":
    NWB_DIR = r'D:\OmissionAnalysis\reconstructed_nwbdata'
    OUTPUT_DIR = r'D:\OmissionAnalysis'
    
    nwb_files = [f for f in os.listdir(NWB_DIR) if f.endswith('.nwb')]
    
    for filename in nwb_files:
        session_id = filename.split('_')[1].split('-')[1]
        nwb_path = os.path.join(NWB_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, f'spikes_by_condition_ses-{session_id}.h5')
        
        extract_spikes_for_session(nwb_path, output_path)
