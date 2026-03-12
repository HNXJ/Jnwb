import sys
import os
import h5py
import numpy as np
import pandas as pd
from scipy.signal import welch
import json
from pynwb import NWBHDF5IO

# Add jnwb to path
sys.path.append(r'D:\jnwb')
from jnwb.analysis import classify_omission_predictability
from jnwb.oglo import extract_good_units

DATA_DIR = r'D:\OmissionAnalysis'
NWB_ROOT = os.path.join(DATA_DIR, 'reconstructed_nwbdata')

def get_session_mapping(nwb_path):
    """Temporary helper to get mapping for classification script."""
    with NWBHDF5IO(nwb_path, 'r', load_namespaces=True) as io:
        nwb = io.read()
        df_elec = nwb.electrodes.to_dataframe()
        probe_col = 'group_name' if 'group_name' in df_elec.columns else 'probe'
        mapping = {}
        for probe, group in df_elec.groupby(probe_col):
            areas = [a for a in group['location'].unique().tolist() if a.lower() not in ['unknown', '']]
            mapping[probe] = areas
        return mapping

def map_idx_to_area(idx, session_mapping):
    if 0 <= idx < 128: probe = 'probeA'; local_idx = idx
    elif 128 <= idx < 256: probe = 'probeB'; local_idx = idx - 128
    elif 256 <= idx < 384: probe = 'probeC'; local_idx = idx - 256
    else: return 'unknown'
    areas = session_mapping.get(probe, [])
    if not areas: return 'unknown'
    n = len(areas)
    if n == 1: return areas[0]
    return areas[min(int(local_idx // (128/n)), n-1)]

def main():
    nwb_files = [f for f in os.listdir(NWB_ROOT) if f.endswith('.nwb')]
    final_results = []

    for filename in nwb_files:
        session_id = filename.split('_')[1].split('-')[1]
        nwb_path = os.path.join(NWB_ROOT, filename)
        spikes_h5 = os.path.join(DATA_DIR, f'spikes_by_condition_ses-{session_id}.h5')
        
        if not os.path.exists(spikes_h5): continue
        print(f"\n--- Batch Classifying Session: {session_id} ---")
        
        # Get mapping
        mapping = get_session_mapping(nwb_path)
        
        with h5py.File(spikes_h5, 'r') as f:
            if 'AAAX' not in f: continue
            
            # Predictable (1) vs Random (0)
            pred_spikes = f['AAAX/spiking_activity'][()]
            rand_list = []
            if 'RXRR' in f: rand_list.append(f['RXRR/spiking_activity'][()])
            if 'AXAB' in f: rand_list.append(f['AXAB/spiking_activity'][()])
            if not rand_list: continue
            rand_spikes = np.vstack(rand_list)
            
            # Extract unit IDs from first group (assuming they match across groups)
            unit_ids = f['AAAX/spiking_activity'].attrs.get('unit_ids', list(range(pred_spikes.shape[1])))
            
            # Group trials by Area
            areas_in_session = set()
            for u_idx in range(pred_spikes.shape[1]):
                areas_in_session.add(map_idx_to_area(u_idx, mapping))
            
            for area in sorted(list(areas_in_session)):
                if area == 'unknown': continue
                print(f"  Area: {area}")
                
                # Get indices for units in this area
                area_unit_indices = [i for i in range(pred_spikes.shape[1]) if map_idx_to_area(i, mapping) == area]
                if not area_unit_indices: continue
                
                # Extract Mean Firing Rate in the Omission Window (3000ms - 4000ms)
                # t=0 is fixation -500ms? No, our H5 window starts at fixation -1000ms.
                # If Stim 1 is at 0ms (relative to NWB start), and fixation is at -500ms...
                # Our 6000ms array: Index 0 = -1000ms, Index 500 = -500ms (Fixation), Index 1000 = 0ms (Stim 1)
                # Therefore, 3000ms-4000ms post-Stim 1 is Index 4000 to 5000.
                idx_start, idx_end = 4000, 5000
                
                X_pred = np.mean(pred_spikes[:, area_unit_indices, idx_start:idx_end], axis=2)
                X_rand = np.mean(rand_spikes[:, area_unit_indices, idx_start:idx_end], axis=2)
                
                X = np.vstack([X_pred, X_rand])
                y = np.concatenate([np.ones(X_pred.shape[0]), np.zeros(X_rand.shape[0])])
                
                res = classify_omission_predictability(X, y)
                if 'error' not in res:
                    final_results.append({
                        "session_id": session_id,
                        "area": area,
                        "accuracy": res['mean_accuracy'],
                        "std": res['std_accuracy'],
                        "n_units": len(area_unit_indices)
                    })
                    print(f"    Accuracy: {res['mean_accuracy']:.2f}")

    with open(os.path.join(DATA_DIR, 'batch_classification_results.json'), 'w') as f:
        json.dump(final_results, f, indent=4)
    print("\nBatch classification results saved.")

if __name__ == "__main__":
    main()
