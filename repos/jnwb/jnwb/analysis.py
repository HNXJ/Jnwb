"""
analysis.py: Refined neural analysis with multi-band spectral support and robust statistics.
Added: Smoothing, Clustering (PCA/UMAP), and Enhanced NaN handling.
"""
import numpy as np
from scipy.signal import welch, coherence, spectrogram
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr
import mne
import pandas as pd

# --- Spectral Band Definitions ---
BANDS = {
    'theta': (3, 6),
    'alpha': (8, 12),
    'low-beta': (13, 20),
    'high-beta': (21, 30),
    'gamma': (35, 70),
    'high-gamma': (75, 150)
}

def smooth_signal(data, sigma=50):
    """Applies Gaussian smoothing to 1D, 2D, or 3D arrays along the last axis."""
    return gaussian_filter1d(data, sigma=sigma, axis=-1)

def get_band_power(data, fs, band_name):
    """Calculates mean power in a specific frequency band with NaN protection."""
    f_min, f_max = BANDS[band_name]
    f, Pxx = welch(data, fs=fs, nperseg=min(len(data), 512))
    mask = (f >= f_min) & (f <= f_max)
    val = np.nanmean(Pxx[mask])
    return np.nan_to_num(val)

def compute_tfr_features(sig, fs):
    """Computes Time-Frequency Representation with robust scaling."""
    freqs = np.linspace(3, 150, 50)
    n_cycles = freqs / 2.
    data = np.nan_to_num(sig)[np.newaxis, np.newaxis, :]
    power = mne.time_frequency.tfr_array_morlet(
        data, sfreq=fs, freqs=freqs, n_cycles=n_cycles, 
        output='power', n_jobs=1, verbose=False
    )
    return freqs, np.nan_to_num(power[0, 0, :, :])

def compute_trial_tfr_dynamics(data_3d, fs, band_name, n_cycles_range=(7, 70)):
    """Computes power dynamics with non-linear cycle scaling and SEM."""
    data_3d = np.nan_to_num(data_3d)
    if data_3d.ndim == 3:
        data_3d = np.nanmean(data_3d, axis=1) 
        
    n_trials, n_time = data_3d.shape
    f_min, f_max = BANDS[band_name]
    freqs = np.linspace(f_min, f_max, 5)
    n_cycles = 7 + (70 - 7) * (freqs - 3) / (100 - 3)
    
    mne_data = data_3d[:, np.newaxis, :]
    power = mne.time_frequency.tfr_array_morlet(
        mne_data, sfreq=fs, freqs=freqs, n_cycles=n_cycles,
        output='power', n_jobs=1, verbose=False
    )
    
    band_power_trials = np.nanmean(power[:, 0, :, :], axis=1) 
    mean_dyn = np.nanmean(band_power_trials, axis=0)
    sem_dyn = np.nanstd(band_power_trials, axis=0) / np.sqrt(n_trials)
    
    return np.nan_to_num(mean_dyn), np.nan_to_num(sem_dyn)

def compute_variability_quenching(data_3d):
    """Calculates trial-to-trial variance with NaN protection."""
    if data_3d.ndim == 3:
        data_3d = np.nanmean(data_3d, axis=1)
    return np.nan_to_num(np.nanvar(data_3d, axis=0))

def compute_pev(data_3d, labels):
    """Computes Percentage of Explained Variance (PEV)."""
    n_trials, n_feats, n_time = data_3d.shape
    unique_labels = np.unique(labels)
    pev_array = np.zeros((n_feats, n_time))
    
    for f in range(n_feats):
        for t in range(n_time):
            y = data_3d[:, f, t]
            var_total = np.nanvar(y)
            if var_total <= 1e-10: continue
            
            ss_error = 0
            for label in unique_labels:
                group = y[labels == label]
                if len(group) > 0:
                    ss_error += np.nansum((group - np.nanmean(group))**2)
            
            var_error = ss_error / n_trials
            pev_array[f, t] = (var_total - var_error) / var_total
            
    return np.nan_to_num(pev_array)

def compute_pca_umap(data_matrix, n_components=3):
    """
    Performs PCA and UMAP for dimensionality reduction.
    data_matrix: (n_samples, n_features)
    """
    from sklearn.decomposition import PCA
    try:
        import umap
        has_umap = True
    except ImportError:
        has_umap = False

    pca = PCA(n_components=n_components)
    pca_results = pca.fit_transform(np.nan_to_num(data_matrix))
    
    umap_results = None
    if has_umap:
        reducer = umap.UMAP(n_components=n_components)
        umap_results = reducer.fit_transform(np.nan_to_num(data_matrix))
        
    return pca_results, umap_results, pca.explained_variance_ratio_
