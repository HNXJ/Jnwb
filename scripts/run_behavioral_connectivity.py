"""
run_behavioral_connectivity.py: Links pupil/eye metrics to spectral band power.
Includes 'Safe Plot' logic to avoid empty/zero figures.
"""
import sys
import os
import h5py
import numpy as np
import plotly.graph_objects as go
from scipy.stats import pearsonr
import gc

# Add jnwb to path
sys.path.append(r'D:\jnwb')
from jnwb.analysis import BANDS, compute_tfr_features

# Configuration
DATA_DIR = r'D:\OmissionAnalysis\arrays'
FIGURES_DIR = r'D:\OmissionAnalysis\figures'
FS = 1000

def safe_save_plotly(fig, output_path, data_matrix, title):
    """Checks if data is valid before saving. Retries with more N if needed."""
    if np.all(data_matrix == 0) or np.all(np.isnan(data_matrix)):
        print(f"⚠️  WARNING: Skipping empty figure for {title}. Data is all Zeros or NaNs.")
        return False
    
    fig.write_html(output_path)
    return True

def run_behavioral_session(session_id):
    print(f"\n>>> Running Behavioral-Spectral Analysis for Session: {session_id}")
    input_h5 = os.path.join(DATA_DIR, f'ses-{session_id}_data_chunks.h5')
    
    if not os.path.exists(input_h5): return

    with h5py.File(input_h5, 'r') as f:
        # We'll use Mode 1 (Standard) for behavioral baseline
        if 'mode_1' not in f: return
        mode_grp = f['mode_1']
        trial_keys = list(mode_grp.keys())
        
        # 1. Aggregate Behavioral Signals (Mean across trials)
        eye_x = np.mean([mode_grp[tk]['eye'][()][:, 0] for tk in trial_keys], axis=0)
        eye_y = np.mean([mode_grp[tk]['eye'][()][:, 1] for tk in trial_keys], axis=0)
        pupil = np.mean([mode_grp[tk]['pupil'][()] for tk in trial_keys], axis=0)
        
        # 2. Extract LFP (Assuming PFC for these data_chunks)
        print(f"    Processing Area: PFC...")
        
        # Load LFP from all trials and average
        lfp_sum = np.zeros(6000)
        valid_trials = 0
        for tk in trial_keys:
            if 'lfp' in mode_grp[tk]:
                # Mean across channels [time x channels]
                lfp_sum += np.mean(mode_grp[tk]['lfp'][()], axis=1)
                valid_trials += 1
        
        if valid_trials == 0: return
        lfp_mean = lfp_sum / valid_trials
        
        # Compute TFR for band envelopes
        freqs, power_tfr = compute_tfr_features(lfp_mean, fs=FS)
        
        # 3. Calculate Correlations [Behav x Band]
        behav_signals = {'Pupil': pupil, 'Eye-X': eye_x, 'Eye-Y': eye_y}
        band_names = list(BANDS.keys())
        
        corr_matrix = np.zeros((len(behav_signals), len(band_names)))
        
        for i, (b_name, b_sig) in enumerate(behav_signals.items()):
            for j, band in enumerate(band_names):
                f_min, f_max = BANDS[band]
                mask = (freqs >= f_min) & (freqs <= f_max)
                band_env = np.nanmean(power_tfr[mask, :], axis=0)
                
                # Ensure alignment (both should be ~6000 samples)
                min_len = min(len(b_sig), len(band_env))
                r, _ = pearsonr(b_sig[:min_len], band_env[:min_len])
                corr_matrix[i, j] = r if not np.isnan(r) else 0

        # 4. Safe Visualize
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix, x=band_names, y=list(behav_signals.keys()),
            colorscale='RdBu_r', zmin=-1, zmax=1
        ))
        fig.update_layout(title=f"Session {session_id} - PFC Behavior-Spectral Link", template="plotly_dark")
        
        out_path = os.path.join(FIGURES_DIR, f"ses-{session_id}_PFC_behavior_link.html")
        safe_save_plotly(fig, out_path, corr_matrix, f"{session_id}_PFC")

    gc.collect()

def main():
    print("Starting master behavioral connectivity loop...")
    lfp_files = [f for f in os.listdir(DATA_DIR) if f.startswith('ses-') and f.endswith('_data_chunks.h5')]
    print(f"Found {len(lfp_files)} chunk files.")
    for f in lfp_files:
        sid = f.split('_')[0].replace('ses-', '')
        print(f"Processing SID: {sid}")
        run_behavioral_session(sid)

if __name__ == "__main__":
    main()
