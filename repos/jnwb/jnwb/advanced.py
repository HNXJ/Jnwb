"""
advanced.py: Advanced cortical analysis including Laminar Alignment and CSD.
Optimized for the jnwb package ecosystem.
"""

import numpy as np
from scipy.ndimage import median_filter
from scipy.interpolate import interp1d

def compute_csd(
    lfp_profile: np.ndarray, 
    spacing_mm: float = 0.04, 
    conductivity: float = 0.3
) -> np.ndarray:
    """
    Computes Current Source Density (CSD) from a 1D spatial LFP profile using
    the second spatial derivative.
    
    Formula: CSD(z) = -sigma * [phi(z+h) - 2*phi(z) + phi(z-h)] / h^2
    
    Args:
        lfp_profile (np.ndarray): 1D array of LFP potentials across depth (channels).
        spacing_mm (float): Distance between adjacent electrodes in mm (e.g., 0.04 for 40um).
        conductivity (float): Extracellular conductivity (sigma).
        
    Returns:
        np.ndarray: The computed CSD profile. Length is N-2 (edges lost in derivative).
    """
    if len(lfp_profile) < 3:
        raise ValueError("LFP profile must have at least 3 channels for CSD.")
        
    h_squared = spacing_mm ** 2
    
    # Calculate the second spatial derivative
    # Note: numpy.diff(x, n=2) computes exactly x[i+2] - 2x[i+1] + x[i]
    second_deriv = np.diff(lfp_profile, n=2)
    
    # CSD
    csd = -conductivity * (second_deriv / h_squared)
    
    return csd

def preprocess_laminar_lfp(
    lfp_matrix: np.ndarray, 
    bad_channels: list = None
) -> np.ndarray:
    """
    Preprocesses LFP for laminar analysis by interpolating bad channels 
    and applying a spatial median filter to smooth artifacts.
    
    Args:
        lfp_matrix (np.ndarray): Shape (n_channels, n_timepoints).
        bad_channels (list): Indices of channels to interpolate.
        
    Returns:
        np.ndarray: Cleaned LFP matrix.
    """
    clean_lfp = lfp_matrix.copy()
    n_chans, n_time = clean_lfp.shape
    
    # 1. Interpolate Bad Channels spatially
    if bad_channels:
        good_channels = [i for i in range(n_chans) if i not in bad_channels]
        if len(good_channels) < 2:
            return clean_lfp # Not enough data to interpolate
            
        for t in range(n_time):
            # Create interpolator for this time slice
            f_interp = interp1d(
                good_channels, 
                clean_lfp[good_channels, t], 
                kind='linear', 
                fill_value='extrapolate'
            )
            clean_lfp[bad_channels, t] = f_interp(bad_channels)
            
    # 2. Spatial Median Filter (size=3) to remove local spiking artifacts from LFP
    # We apply it across the channel dimension (axis 0)
    filtered_lfp = median_filter(clean_lfp, size=(3, 1))
    
    return filtered_lfp
