"""
analysis.py: Expanded neural analysis with multi-band spectral support.
"""
import numpy as np
from scipy.signal import welch, coherence, spectrogram
from scipy.stats import pearsonr
import mne

# --- Spectral Band Definitions ---
BANDS = {
    'theta': (3, 6),
    'alpha': (8, 12),
    'low-beta': (13, 20),
    'high-beta': (21, 30),
    'gamma': (35, 70),
    'high-gamma': (75, 150)
}

def get_band_power(data, fs, band_name):
    """Calculates mean power in a specific frequency band."""
    f_min, f_max = BANDS[band_name]
    f, Pxx = welch(data, fs=fs, nperseg=512)
    mask = (f >= f_min) & (f <= f_max)
    # Ignore NaNs during mean calculation
    return np.nanmean(Pxx[mask])

def compute_cross_band_correlation(sig1, sig2, fs):
    """
    Calculates power correlation between all bands of two signals.
    """
    results = {}
    for b1 in BANDS:
        for b2 in BANDS:
            # Note: This is a static correlation. 
            # For time-resolved, we would use the TFR below.
            p1 = get_band_power(sig1, fs, b1)
            p2 = get_band_power(sig2, fs, b2)
            results[f"{b1}_{b2}"] = (p1, p2)
    return results

def compute_tfr_features(sig, fs):
    """
    Computes Time-Frequency Representation for band mapping.
    """
    freqs = np.linspace(3, 150, 50)
    n_cycles = freqs / 2.
    # mne expects [epochs, channels, times]
    data = sig[np.newaxis, np.newaxis, :]
    power = mne.time_frequency.tfr_array_morlet(
        data, sfreq=fs, freqs=freqs, n_cycles=n_cycles, 
        output='power', n_jobs=1, verbose=False
    )
    return freqs, power[0, 0, :, :] # (n_freqs, n_time)

def compute_trial_tfr_dynamics(data_3d, fs, band_name):
    """
    Computes power dynamics for each trial in a specific band.
    data_3d: (trials, time) or (trials, channels, time)
    Returns: mean_power, sem_power (over trials)
    """
    if data_3d.ndim == 3: # (trials, chans, time)
        data_3d = np.mean(data_3d, axis=1) # Mean over channels
        
    n_trials, n_time = data_3d.shape
    f_min, f_max = BANDS[band_name]
    
    # Use Morlet for trial-level precision
    freqs = np.linspace(f_min, f_max, 5)
    n_cycles = freqs / 2.
    
    # mne expects [epochs, channels, times]
    mne_data = data_3d[:, np.newaxis, :]
    power = mne.time_frequency.tfr_array_morlet(
        mne_data, sfreq=fs, freqs=freqs, n_cycles=n_cycles,
        output='power', n_jobs=1, verbose=False
    ) # (trials, 1, freqs, time)
    
    # Average across frequencies in the band
    band_power_trials = np.nanmean(power[:, 0, :, :], axis=1) # (trials, time)
    
    mean_dyn = np.nanmean(band_power_trials, axis=0)
    sem_dyn = np.nanstd(band_power_trials, axis=0) / np.sqrt(n_trials)
    
    return mean_dyn, sem_dyn

def compute_variability_quenching(data_3d):
    """
    Calculates trial-to-trial variance over time.
    """
    if data_3d.ndim == 3:
        data_3d = np.mean(data_3d, axis=1)
    
    # Variance across trials at each time point
    return np.nanvar(data_3d, axis=0)

