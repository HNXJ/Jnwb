"""
run_multiband_analysis.py: Master script for cross-band and spike-band spectral connectivity.
Part of the jnwb research suite.
"""
import sys
import os
import h5py
import numpy as np
import plotly.graph_objects as go
import gc

# Add jnwb to path
sys.path.append(r'D:\jnwb')
from jnwb.analysis import compute_tfr_features, correlate_spike_to_bands, BANDS

# Configuration
DATA_DIR = r'D:\OmissionAnalysis'
LFP_DIR = DATA_DIR 
SPIKE_DIR = os.path.join(DATA_DIR, 'arrays')
FIGURES_DIR = os.path.join(DATA_DIR, 'figures')
FS = 1000

# GPU Acceleration Detection
try:
    import cupy as cp
    HAS_CUDA = True
    print("🚀 NVIDIA CUDA detected (A4000). Enabling GPU acceleration.")
except ImportError:
    HAS_CUDA = False
    print("ℹ️ CUDA not detected. Using CPU (NumPy).")

def run_session_multiband(session_id):
    print(f"\n>>> Running Multi-Band Spectral Analysis for Session: {session_id}")
    lfp_path = os.path.join(LFP_DIR, f'lfp_by_area_ses-{session_id}.h5')
    spike_path = os.path.join(DATA_DIR, f'spikes_by_condition_ses-{session_id}.h5') # Root check
    
    if not os.path.exists(lfp_path):
        lfp_path = os.path.join(DATA_DIR, f'lfp_by_area_ses-{session_id}.h5') # Fallback to root
    
    if not os.path.exists(spike_path):
        spike_path = os.path.join(DATA_DIR, f'arrays/spikes_by_condition_ses-{session_id}.h5') # Fallback to arrays

    with h5py.File(lfp_path, 'r') as hl, h5py.File(spike_path, 'r') as hs:
        areas = sorted(list(hl.keys()))
        band_names = list(BANDS.keys())
        
        for area in areas:
            print(f"    Processing Area: {area}...")
            
            # Target: AAAX Omission Condition
            if 'AAAX' not in hl[area]: continue
            
            # 1. Load Mean LFP and Spikes for the area
            # Average across trials and channels for spectral signature
            lfp_mean = np.mean(hl[f"{area}/AAAX"][()], axis=(0, 1))
            
            # Get spike density for the same condition
            if 'AAAX' in hs:
                # Average across trials and all units in this area (simplified for MVP)
                spikes_pop = np.mean(np.mean(hs['AAAX/spiking_activity'][()], axis=0), axis=0)
            else:
                spikes_pop = np.zeros_like(lfp_mean)

            # 2. Compute TFR
            freqs, power_tfr = compute_tfr_features(lfp_mean, fs=FS)
            
            # 3. Cross-Band Correlation Matrix
            # (Correlating the temporal envelopes of each band)
            band_envelopes = []
            for band, (f_min, f_max) in BANDS.items():
                mask = (freqs >= f_min) & (freqs <= f_max)
                env = np.nanmean(power_tfr[mask, :], axis=0)
                band_envelopes.append(env)
            
            corr_matrix = np.corrcoef(np.array(band_envelopes))
            # Handle NaNs in correlation
            corr_matrix = np.nan_to_num(corr_matrix)

            # 4. Spike-Band Correlation
            spike_corr = correlate_spike_to_bands(spikes_pop, power_tfr, freqs)
            spike_vals = [spike_corr.get(b, 0) for b in band_names]

            # 5. Visualize Matrix (Cross-Band)
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix, x=band_names, y=band_names, 
                colorscale='Viridis', zmin=-1, zmax=1
            ))
            fig.update_layout(title=f"Session {session_id} - {area} Cross-Band Correlation (AAAX)", template="plotly_dark")
            fig.write_html(os.path.join(FIGURES_DIR, f"ses-{session_id}_{area}_band_corr.html"))

            # 6. Visualize Bar Chart (Spike-Band)
            fig_spike = go.Figure(data=go.Bar(x=band_names, y=spike_vals, marker_color='#CFB87C'))
            fig_spike.update_layout(title=f"Session {session_id} - {area} Spike-Band Coupling", template="plotly_dark")
            fig_spike.write_html(os.path.join(FIGURES_DIR, f"ses-{session_id}_{area}_spike_band.html"))

    gc.collect()

def main():
    print("Starting master multi-band loop...")
    os.makedirs(FIGURES_DIR, exist_ok=True)
    lfp_files = [f for f in os.listdir(LFP_DIR) if f.endswith('.h5')]
    print(f"Found {len(lfp_files)} LFP files.")
    for f in lfp_files:
        sid = f.split('_')[-1].split('.')[0].replace('ses-', '')
        print(f"Processing Multi-Band SID: {sid}")
        run_session_multiband(sid)

if __name__ == "__main__":
    main()
