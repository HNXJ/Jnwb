"""
analysis.py: Modular Python implementations of core neural analysis functions.
Optimized for the jnwb package ecosystem.
"""

import numpy as np
from scipy.signal import csd, welch
from typing import Tuple, Optional

def compute_spike_field_coherence(
    lfp: np.ndarray, 
    spk: np.ndarray, 
    fs: float, 
    f_range: Tuple[float, float],
    n_freqs: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate Spike-Field Coherence (SFC) for multiple neuron-channel pairs.
    lfp: (n_trials, n_channels, n_time)
    spk: (n_trials, n_neurons, n_time)
    Returns: (coherence, freqs)
    """
    n_trials, n_channels, n_time = lfp.shape
    trials2, n_neurons, time2 = spk.shape

    if n_trials != trials2 or n_time != time2:
        raise ValueError("Dimensions of LFP and SPK must match.")

    f_min, f_max = f_range
    # Use CSD to calculate coherence
    # We'll calculate mean across trials for better estimation
    
    # Calculate cross-spectral density and power spectral densities
    # We'll pick a window size for welch (e.g., 512 samples)
    nperseg = min(n_time, 512)
    
    # Pre-allocate output [Neurons x Channels x Frequency]
    # We first find the frequency vector length
    _, temp_f = welch(lfp[0, 0, :], fs=fs, nperseg=nperseg)
    freq_mask = (temp_f >= f_min) & (temp_f <= f_max)
    actual_freqs = temp_f[freq_mask]
    
    coherence_matrix = np.zeros((n_neurons, n_channels, len(actual_freqs)))

    for n_idx in range(n_neurons):
        for c_idx in range(n_channels):
            # Calculate mean CSD and PSD across trials
            s_xy_total = 0
            s_xx_total = 0
            s_yy_total = 0
            
            for t_idx in range(n_trials):
                f, s_xy = csd(lfp[t_idx, c_idx, :], spk[t_idx, n_idx, :], fs=fs, nperseg=nperseg)
                _, s_xx = welch(lfp[t_idx, c_idx, :], fs=fs, nperseg=nperseg)
                _, s_yy = welch(spk[t_idx, n_idx, :], fs=fs, nperseg=nperseg)
                
                s_xy_total += s_xy[freq_mask]
                s_xx_total += s_xx[freq_mask]
                s_yy_total += s_yy[freq_mask]
            
            # Coherence = |mean(Sxy)|^2 / (mean(Sxx) * mean(Syy))
            coherence = (np.abs(s_xy_total/n_trials)**2) / ((s_xx_total/n_trials) * (s_yy_total/n_trials) + 1e-10)
            coherence_matrix[n_idx, c_idx, :] = coherence

    return coherence_matrix, actual_freqs

def compute_percent_explained_variance(
    data: np.ndarray, 
    group_labels: np.ndarray, 
    use_omega_squared: bool = True
) -> dict:
    """
    Calculates PEV (Eta-squared or Omega-squared) for neural data.
    data: (n_observations, n_variables)
    group_labels: (n_observations,)
    """
    n_total = len(group_labels)
    unique_groups = np.unique(group_labels)
    n_groups = len(unique_groups)
    
    if n_groups < 2:
        return {"error": "At least 2 groups required for ANOVA/PEV"}

    # Grand mean
    grand_mean = np.mean(data, axis=0)
    
    # Sum of Squares Total (SStotal)
    ss_total = np.sum((data - grand_mean)**2, axis=0)
    
    # Sum of Squares Groups (SSgrps)
    ss_grps = np.zeros(data.shape[1])
    for g in unique_groups:
        group_data = data[group_labels == g]
        n_g = len(group_data)
        group_mean = np.mean(group_data, axis=0)
        ss_grps += n_g * (group_mean - grand_mean)**2
        
    if use_omega_squared:
        # Omega-squared = (SSgrps - (df_grps * MSerr)) / (SStotal + MSerr)
        df_grps = n_groups - 1
        df_err = n_total - n_groups
        ms_err = (ss_total - ss_grps) / df_err
        
        pev = (ss_grps - (df_grps * ms_err)) / (ss_total + ms_err + 1e-10)
    else:
        # Eta-squared = SSgrps / SStotal
        pev = ss_grps / (ss_total + 1e-10)
        
    return {"pev": pev, "groups": unique_groups, "n_total": n_total}

def classify_omission_predictability(
    features: np.ndarray, 
    labels: np.ndarray, 
    cv_folds: int = 5
) -> dict:
    """
    Trains a Random Forest classifier to distinguish between Predictable (AAAx) 
    and Random (RXRR/AXAB) omissions based on neural features (e.g., spectral power, firing rate).
    
    Args:
        features: (n_trials, n_features) array of neural data during the omission window.
        labels: (n_trials,) array of binary labels (1 = Predictable, 0 = Random).
        cv_folds: Number of cross-validation folds.
        
    Returns:
        dict: Classification metrics including mean accuracy, std, and feature importances.
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        import warnings
        
        # Suppress future warnings from sklearn
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            scores = cross_val_score(clf, features, labels, cv=cv_folds, scoring='accuracy')
            
            # Fit on all data to get feature importances
            clf.fit(features, labels)
            importances = clf.feature_importances_
            
            return {
                "mean_accuracy": float(np.mean(scores)),
                "std_accuracy": float(np.std(scores)),
                "feature_importances": importances.tolist(),
                "n_predictable": int(np.sum(labels == 1)),
                "n_random": int(np.sum(labels == 0))
            }
    except ImportError:
        return {"error": "scikit-learn is not installed. Run: pip install scikit-learn"}
