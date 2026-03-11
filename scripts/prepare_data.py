import h5py
import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO
import os
import gc
import sys

PRE_T = 1.0
POST_T = 5.0
TOTAL_T = PRE_T + POST_T

def extract_timeseries_chunk(timeseries, start_time, duration):
    if timeseries.rate is not None:
        fs = timeseries.rate
        t0 = timeseries.starting_time if timeseries.starting_time is not None else 0.0
        start_idx = int((start_time - t0) * fs)
        n_samples = int(duration * fs)
    else:
        ts = timeseries.timestamps[()]
        start_idx = np.searchsorted(ts, start_time)
        end_idx = np.searchsorted(ts, start_time + duration)
        n_samples = end_idx - start_idx
    
    if start_idx < 0: return None
    try:
        return timeseries.data[start_idx : start_idx + n_samples]
    except: return None

def main(session_id, nwb_path):
    output_h5 = f'D:/hnxj-gemini/ses-{session_id}_data_chunks.h5'
    trials_path = f'D:/hnxj-gemini/ses-{session_id}_trials.csv'
    
    if os.path.exists(output_h5):
        print(f"Chunks for {session_id} already exist. Skipping.")
        return

    df = pd.read_csv(trials_path)
    trials = df[df['correct'] == 1].drop_duplicates(subset=['trial_num']).sort_values('start_time')
    
    print(f"Extracting {len(trials)} trials for {session_id}...")
    
    with h5py.File(output_h5, 'a') as h5:
        with NWBHDF5IO(nwb_path, 'r', load_namespaces=True) as io:
            nwb = io.read()
            eye = nwb.acquisition['eye_1_tracking']
            pupil = nwb.acquisition['pupil_1_tracking']
            photo = nwb.acquisition['photodiode_1_tracking']
            lfp_pfc = nwb.acquisition['probe_2_lfp']
            
            for i, (_, row) in enumerate(trials.iterrows()):
                trial_id = int(row['trial_num'])
                mode_id = int(row['task_condition_number'])
                t_start = row['start_time'] - PRE_T
                
                grp = h5.create_group(f"mode_{mode_id}/trial_{trial_id}")
                grp.create_dataset('eye', data=extract_timeseries_chunk(eye, t_start, TOTAL_T), compression='gzip')
                grp.create_dataset('pupil', data=extract_timeseries_chunk(pupil, t_start, TOTAL_T), compression='gzip')
                grp.create_dataset('photodiode', data=extract_timeseries_chunk(photo, t_start, TOTAL_T), compression='gzip')
                grp.create_dataset('lfp', data=extract_timeseries_chunk(lfp_pfc, t_start, TOTAL_T), compression='gzip')
                
                spike_grp = grp.create_group('spikes')
                for u_idx in range(len(nwb.units)):
                    u_times = nwb.units.get_unit_spike_times(u_idx)
                    mask = (u_times >= row['start_time'] - PRE_T) & (u_times <= row['start_time'] + POST_T)
                    rel_times = u_times[mask] - row['start_time']
                    if len(rel_times) > 0:
                        spike_grp.create_dataset(str(u_idx), data=rel_times, compression='gzip')
                
                if (i+1) % 10 == 0:
                    print(f"  {session_id}: Processed {i+1}/{len(trials)} trials.")
                    h5.flush()
                    gc.collect()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python prepare_data_chunks.py <session_id> <nwb_path>")
    else:
        main(sys.argv[1], sys.argv[2])
