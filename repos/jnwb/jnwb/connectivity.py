"""
connectivity.py: Multi-scale functional connectivity analysis.
Includes Time-Frequency Coherence and Cross-Signal Correlations.
"""
import numpy as np
import pandas as pd
from scipy.signal import coherence, correlate
from typing import Dict, List, Optional

def compute_tfr_correlation(tfr_data1: np.ndarray, tfr_data2: np.ndarray):
    """
    Calculates the correlation between two TFR datasets.
    tfr_data: (n_freqs, n_time)
    """
    # Flatten and correlate for each frequency or across all
    correlations = []
    for f in range(tfr_data1.shape[0]):
        c = np.corrcoef(tfr_data1[f, :], tfr_data2[f, :])[0, 1]
        correlations.append(c)
    return np.array(correlations)

def compute_sfc(spikes_3d, lfp_3d, fs):
    """
    Computes Spike-Field Coherence (SFC) across trials.
    Input: (trials, time) for both
    Returns: freq, coherence
    """
    from scipy.signal import coherence
    # Flatten across trials for global coherence or average?
    # Usually coherence is calculated on the concatenated trials or averaged.
    # We will average the cross-spectral densities.
    f, Cxy = coherence(spikes_3d.flatten(), lfp_3d.flatten(), fs=fs, nperseg=1024)
    return f, Cxy

def compute_granger_causality(sig1, sig2, max_lag_ms=100, fs=1000):
    """
    Computes a simple bivariate Granger Causality score.
    max_lag_ms: How far back to look for predictability (up to 100ms).
    """
    from statsmodels.tsa.stattools import grangercausalitytests
    import pandas as pd
    
    # maxlag in samples
    max_lag = int(max_lag_ms * (fs / 1000))
    data = pd.DataFrame({'s1': sig1, 's2': sig2})
    
    # Test if s2 granger-causes s1
    res = grangercausalitytests(data[['s1', 's2']], maxlag=[max_lag], verbose=False)
    # Return 1 - p_value as a 'causality score'
    return 1 - res[max_lag][0]['ssr_ftest'][1]

def compute_granger_causality(sig1, sig2, max_lag_ms=100, fs=1000):
    """
    Computes a simple bivariate Granger Causality score.
    max_lag_ms: How far back to look for predictability (up to 100ms).
    """
    from statsmodels.tsa.stattools import grangercausalitytests
    import pandas as pd
    
    # maxlag in samples
    max_lag = int(max_lag_ms * (fs / 1000))
    data = pd.DataFrame({'s1': sig1, 's2': sig2})
    
    # Test if s2 granger-causes s1
    res = grangercausalitytests(data[['s1', 's2']], maxlag=[max_lag], verbose=False)
    # Return 1 - p_value as a 'causality score'
    return 1 - res[max_lag][0]['ssr_ftest'][1]

def compute_plv(sig1, sig2):
    """
    Computes Phase-Locking Value (PLV) between two signals.
    """
    from scipy.signal import hilbert
    p1 = np.angle(hilbert(sig1))
    p2 = np.angle(hilbert(sig2))
    return np.abs(np.mean(np.exp(1j * (p1 - p2))))

