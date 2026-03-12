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

def compute_signal_sync(sig1: np.ndarray, sig2: np.ndarray, fs: float):
    """
    Computes spectral coherence between two simultaneous signals.
    sig: (n_time,)
    """
    f, Cxy = coherence(sig1, sig2, fs=fs, nperseg=256)
    return f, Cxy

def identify_context_networks(connectivity_matrices: Dict[str, np.ndarray]):
    """
    (TBD) Identifies active subnetworks based on connectivity patterns
    across different task contexts (Fixation, Stim, Omission).
    """
    pass
