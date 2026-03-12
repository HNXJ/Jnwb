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

def compute_trial_tfr_dynamics(data_3d, fs, band_name, n_cycles_range=(7, 70)):
    """
    Computes power dynamics for each trial in a specific band using precision parameters.
    cycles: Scaled from n_cycles_range[0] at 3Hz to n_cycles_range[1] at 100Hz.
    """
    if data_3d.ndim == 3: # (trials, chans, time)
        data_3d = np.nanmean(data_3d, axis=1) 
        
    n_trials, n_time = data_3d.shape
    f_min, f_max = BANDS[band_name]
    
    freqs = np.linspace(f_min, f_max, 5)
    # Linear scaling of cycles based on frequency
    # cycles = c_min + (c_max - c_min) * (f - f_min) / (f_total_max - f_total_min)
    n_cycles = 7 + (70 - 7) * (freqs - 3) / (100 - 3)
    
    mne_data = data_3d[:, np.newaxis, :]
    power = mne.time_frequency.tfr_array_morlet(
        mne_data, sfreq=fs, freqs=freqs, n_cycles=n_cycles,
        output='power', n_jobs=1, verbose=False
    )
    
    band_power_trials = np.nanmean(power[:, 0, :, :], axis=1) 
    
    mean_dyn = np.nanmean(band_power_trials, axis=0)
    sem_dyn = np.nanstd(band_power_trials, axis=0) / np.sqrt(n_trials)
    
    return mean_dyn, sem_dyn

def compute_variability_quenching(data_3d):
    """
    Calculates trial-to-trial variance over time.
    """
    if data_3d.ndim == 3:
        data_3d = np.nanmean(data_3d, axis=1)
    return np.nanvar(data_3d, axis=0)

def compute_pev(data_3d, labels):
    """
    Computes Percentage of Explained Variance (PEV).
    data_3d: (trials, features, time)
    """
    n_trials, n_feats, n_time = data_3d.shape
    unique_labels = np.unique(labels)
    pev_array = np.zeros((n_feats, n_time))
    
    for f in range(n_feats):
        for t in range(n_time):
            y = data_3d[:, f, t]
            var_total = np.nanvar(y)
            if var_total == 0: continue
            
            ss_error = 0
            for label in unique_labels:
                group = y[labels == label]
                if len(group) > 0:
                    ss_error += np.nansum((group - np.nanmean(group))**2)
            
            var_error = ss_error / n_trials
            pev_array[f, t] = (var_total - var_error) / var_total
            
    return pev_array

def decode_over_time(X_3d, y, window_size=100, step=50):
    """Sliding window SVM decoding."""
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    
    n_trials, n_feats, n_time = X_3d.shape
    time_points = range(0, n_time - window_size, step)
    accuracies = []
    
    for t in time_points:
        X_win = np.nanmean(X_3d[:, :, t : t + window_size], axis=2)
        score = np.mean(cross_val_score(SVC(kernel='linear'), X_win, y, cv=3))
        accuracies.append(score)
        
    return np.array(time_points), np.array(accuracies)
