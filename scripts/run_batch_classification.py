"""
Batch classification of Predictable vs Random omissions across all sessions and brain areas.
"""
import sys
import os
import h5py
import numpy as np
import pandas as pd
from scipy.signal import welch
import json

# Add jnwb to path
sys.path.append(r'D:\jnwb')
from jnwb.analysis import classify_omission_predictability

DATA_DIR = r'D:\OmissionAnalysis'
STATS_PATH = os.path.join(DATA_DIR, 'all_units_stats.json')

def get_power_features(spikes, fs, f_band=(38, 40)):
    """Calculates band power for each unit and trial."""
    n_trials, n_units, n_time = spikes.shape
    features = np.zeros((n_trials, n_units))
    
    for i in range(n_trials):
        for j in range(n_units):
            f, Pxx = welch(spikes[i, j, :], fs=fs, nperseg=256)
            band_mask = (f >= f_band[0]) & (f <= f_band[1])
            features[i, j] = np.mean(Pxx[band_mask])
            
    return features

def run_batch_classification():
    # Load session metadata
    with open(STATS_PATH, 'r') as f:
        meta = json.load(f)
    
    results = []
    
    for session_info in meta['sessions']:
        session_id = session_info['session_id']
        input_h5 = os.path.join(DATA_DIR, f'spikes_by_condition_ses-{session_id}.h5')
        
        if not os.path.exists(input_h5):
            continue
            
        print(f"\n>>> Running Area-Specific Classification for Session: {session_id}")
        
        with h5py.File(input_h5, 'r') as f:
            # 1. Identify Target Groups (Predictable: AAAX, Random: RXRR/AXAB)
            if 'AAAX' not in f: continue
            
            predictable_trials = f['AAAX/spiking_activity'][()]
            
            random_trials_list = []
            if 'RXRR' in f: random_trials_list.append(f['RXRR/spiking_activity'][()])
            if 'AXAB' in f: random_trials_list.append(f['AXAB/spiking_activity'][()])
            
            if not random_trials_list: continue
            random_trials = np.vstack(random_trials_list)
            
            # 2. Get Area Mapping for the units in this session
            # For simplicity, we'll use the areas identified in analyze_all_units.py
            # Since spikes_by_condition already filtered for 'good' units, 
            # we need to map those units specifically.
            
            # Map areas for ALL good units in this session
            areas = ['V1', 'V2', 'MT', 'MST', 'PFC']
            
            for area in areas:
                # Find indices of units belonging to this area (mock logic for now)
                # In full run, we would use the df_good_units from analyze_all_units
                # Here we'll just use a placeholder to demonstrate the area-specific loop
                
                print(f"  Classifying for Area: {area}...")
                
                # Filter predictable/random trials for units in this area
                # (Placeholder: Using all units until area-specific unit indices are integrated)
                all_predictable_features = get_power_features(predictable_trials, fs=1000)
                all_random_features = get_power_features(random_trials, fs=1000)
                
                X = np.vstack([all_predictable_features, all_random_features])
                y = np.concatenate([np.ones(len(all_predictable_features)), np.zeros(len(all_random_features))])
                
                res = classify_omission_predictability(X, y)
                
                if 'error' not in res:
                    results.append({
                        "session_id": session_id,
                        "area": area,
                        "accuracy": res['mean_accuracy'],
                        "std": res['std_accuracy'],
                        "n_predictable": res['n_predictable'],
                        "n_random": res['n_random']
                    })
                    print(f"    Result: {res['mean_accuracy']:.2f}")

    # Save Results
    output_path = os.path.join(DATA_DIR, 'batch_classification_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nBatch classification complete. Results saved to {output_path}")

if __name__ == "__main__":
    run_batch_classification()
