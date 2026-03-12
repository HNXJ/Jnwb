"""
run_connectivity_batch.py: Master pipeline for cross-area and cross-signal connectivity.
Part of the jnwb research suite.
"""
import sys
import os
import h5py
import numpy as np
import plotly.graph_objects as go
from scipy.signal import coherence
from scipy.stats import pearsonr
import gc

# Add jnwb to path
sys.path.append(r'D:\jnwb')
from jnwb.connectivity import compute_tfr_correlation, compute_signal_sync

# Configuration
LFP_DIR = r'D:\OmissionAnalysis'
SPIKE_DIR = r'D:\OmissionAnalysis\arrays'
FIGURES_DIR = r'D:\OmissionAnalysis\figures'
FS = 1000

def run_session_connectivity(session_id):
    print(f"\n>>> Analyzing Connectivity for Session: {session_id}")
    lfp_path = os.path.join(LFP_DIR, f'lfp_by_area_ses-{session_id}.h5')
    spike_path = os.path.join(SPIKE_DIR, f'spikes_by_condition_ses-{session_id}.h5')
    
    if not os.path.exists(lfp_path): return

    with h5py.File(lfp_path, 'r') as hl, h5py.File(spike_path, 'r') as hs:
        areas = sorted(list(hl.keys()))
        print(f"  Areas: {areas}")
        
        # 1. Cross-Area LFP Coherence (Target: AAAX Omission)
        if 'AAAX' in hl[areas[0]]:
            print(f"  Calculating Cross-Area Coherence (AAAX)...")
            matrix = np.zeros((len(areas), len(areas)))
            
            for i, a1 in enumerate(areas):
                for j, a2 in enumerate(areas):
                    if i >= j: continue
                    
                    # Get mean signal across channels for these areas
                    sig1 = np.mean(hl[f"{a1}/AAAX"][()], axis=(0, 1))
                    sig2 = np.mean(hl[f"{a2}/AAAX"][()], axis=(0, 1))
                    
                    f, Cxy = compute_signal_sync(sig1, sig2, fs=FS)
                    # Average coherence in Gamma (38-40Hz)
                    gamma_mask = (f >= 38) & (f <= 40)
                    matrix[i, j] = np.mean(Cxy[gamma_mask])
                    matrix[j, i] = matrix[i, j]

            # Visualize Matrix
            fig = go.Figure(data=go.Heatmap(z=matrix, x=areas, y=areas, colorscale='Viridis'))
            fig.update_layout(title=f"Session {session_id} - Cross-Area Gamma Coherence (Omission)", template="plotly_dark")
            fig.write_html(os.path.join(FIGURES_DIR, f"ses-{session_id}_coherence_matrix.html"))

        # 2. Pupil-PFC Connectivity
        # (Assuming Pupil data was added to the chunks in previous step)
        # Placeholder for Pupil-Gamma correlation logic
        
    gc.collect()

def main():
    print("Starting master connectivity loop...")
    os.makedirs(FIGURES_DIR, exist_ok=True)
    lfp_files = [f for f in os.listdir(LFP_DIR) if f.endswith('.h5')]
    print(f"Found {len(lfp_files)} LFP files: {lfp_files}")
    for f in lfp_files:
        sid = f.split('_')[-1].split('.')[0].replace('ses-', '')
        print(f"Processing SID: {sid}")
        run_session_connectivity(sid)

if __name__ == "__main__":
    main()
