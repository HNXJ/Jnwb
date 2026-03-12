"""
Batch extracts LFP signals by area and condition for all sessions.
"""
import sys
import os
import h5py
import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO
import gc

# Add jnwb to path
sys.path.append(r'D:\jnwb')
from jnwb.lfp import get_lfp_probe_mapping, map_channels_to_areas, extract_lfp_epoch
from jnwb.oglo_v2 import get_oglo_trial_masks_v2 as get_trial_masks

NWB_DIR = r'D:\OmissionAnalysis\reconstructed_nwbdata'
OUTPUT_DIR = r'D:\OmissionAnalysis\LFP_Extractions'

# Timing
PRE_S, POST_S = 1.0, 5.0
DURATION = PRE_S + POST_S

def process_session_lfp(nwb_path, session_id):
    print(f"\n>>> Extracting LFP for Session: {session_id}")
    output_path = os.path.join(OUTPUT_DIR, f'lfp_by_area_ses-{session_id}.h5')
    
    if os.path.exists(output_path):
        print("  Output already exists. Skipping.")
        return

    with NWBHDF5IO(nwb_path, 'r', load_namespaces=True) as io:
        nwb = io.read()
        
        # 1. Map Areas
        probe_mapping = get_lfp_probe_mapping(nwb)
        chan_to_area = map_channels_to_areas(probe_mapping)
        unique_areas = sorted(list(set(chan_to_area.values())))
        print(f"  Areas identified: {unique_areas}")

        # 2. Identify Correct Trials
        trials_df = nwb.intervals['omission_glo_passive'].to_dataframe()
        correct_trials = trials_df[trials_df['correct'].astype(str) == '1.0'].drop_duplicates(subset=['trial_num']).sort_values('start_time')
        trial_masks = get_trial_masks(correct_trials)

        # 3. Access LFP Objects
        # Map probeA/B/C to actual NWB acquisition keys
        lfp_objects = {}
        for probe in probe_mapping.keys():
            key = f'{probe}_lfp'
            if key in nwb.acquisition:
                lfp_objects[probe] = nwb.acquisition[key]

        # 4. Extract and Save by Area/Condition
        with h5py.File(output_path, 'w') as h5:
            for area in unique_areas:
                print(f"    Processing Area: {area}...")
                area_grp = h5.create_group(area)
                
                # Find global channel indices for this area
                # (Assuming standard 128-channel increments)
                area_chans = [c for c, a in chan_to_area.items() if a == area]
                
                for cond_name, mask in trial_masks.items():
                    cond_trials = correct_trials[mask]
                    if len(cond_trials) == 0: continue
                    
                    # Pre-allocate array [trials x area_channels x time]
                    # We'll use 1000Hz sampling
                    n_samples = int(DURATION * 1000)
                    data_array = np.zeros((len(cond_trials), len(area_chans), n_samples), dtype=np.float32)
                    
                    for i, (_, row) in enumerate(cond_trials.iterrows()):
                        t_start = row['start_time'] - PRE_S
                        
                        # We need to extract from the correct probe object
                        # Channel ID 0-127 -> probeA, 128-255 -> probeB, etc.
                        for j, global_cid in enumerate(area_chans):
                            if global_cid < 128: p_key, local_cid = 'probeA', global_cid
                            elif global_cid < 256: p_key, local_cid = 'probeB', global_cid - 128
                            else: p_key, local_cid = 'probeC', global_cid - 256
                            
                            if p_key in lfp_objects:
                                # This is slow but memory safe. 
                                # For speed, we could slice all channels of a probe at once.
                                # Let's optimize: Slice the whole probe's time window once.
                                pass
                        
                        # OPTIMIZED SLICING (per trial, per probe)
                        for p_key, lfp_obj in lfp_objects.items():
                            probe_slice = extract_lfp_epoch(lfp_obj, t_start, DURATION)
                            if probe_slice is not None:
                                # Find which area channels belong to this probe
                                p_start = {'probeA':0, 'probeB':128, 'probeC':256}[p_key]
                                p_end = p_start + 128
                                
                                # Intersect area_chans with this probe's range
                                overlap = [c for c in area_chans if p_start <= c < p_end]
                                for c in overlap:
                                    local_idx = c - p_start
                                    target_idx = area_chans.index(c)
                                    data_array[i, target_idx, :] = probe_slice[:, local_idx]

                    area_grp.create_dataset(cond_name, data=data_array, compression='gzip', chunks=True)
                    print(f"      Saved {cond_name}: {data_array.shape}")
                    gc.collect()

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    nwb_files = [f for f in os.listdir(NWB_DIR) if f.endswith('.nwb')]
    for f in nwb_files:
        sid = f.split('_')[1].split('-')[1]
        process_session_lfp(os.path.join(NWB_DIR, f), sid)
