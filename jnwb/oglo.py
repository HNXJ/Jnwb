import numpy as np
import pandas as pd

def get_trial_masks(df):
    """
    Identifies the 12 'Visual Omission Oddball' (OGLO) trial groups from an NWB interval dataframe.
    
    Args:
        df (pd.DataFrame): DataFrame extracted from nwb.intervals['omission_glo_passive'].
                           Requires columns: 'task_block_number', 'correct', 'stimulus_number', 'is_omission'.
    
    Returns:
        dict: A dictionary mapping condition names (e.g., 'AAAB', 'AAAX') to boolean pandas Series (masks).
    """
    # Base filter for all valid trials
    base_mask = (df['correct'] == 1) & (df['stimulus_number'] == 3)
    
    # Block 2: AAAB Sequence
    aaab = base_mask & (df['task_block_number'] == 2) & df['is_omission'].isna()
    axab = (df['task_block_number'] == 2) & (df['correct'] == 1) & (df['is_omission'] == 1) & (df['stimulus_number'] == 3)
    aaxb = (df['task_block_number'] == 2) & (df['correct'] == 1) & (df['is_omission'] == 1) & (df['stimulus_number'] == 4)
    aaax = (df['task_block_number'] == 2) & (df['correct'] == 1) & (df['is_omission'] == 1) & (df['stimulus_number'] == 5)

    # Block 4: BBBA Sequence
    bbba = base_mask & (df['task_block_number'] == 4) & df['is_omission'].isna()
    bxba = (df['task_block_number'] == 4) & (df['correct'] == 1) & (df['is_omission'] == 1) & (df['stimulus_number'] == 3)
    bbxa = (df['task_block_number'] == 4) & (df['correct'] == 1) & (df['is_omission'] == 1) & (df['stimulus_number'] == 4)
    bbbx = (df['task_block_number'] == 4) & (df['correct'] == 1) & (df['is_omission'] == 1) & (df['stimulus_number'] == 5)

    # Block 5: RRRR Sequence (Randomized)
    rrrr = base_mask & (df['task_block_number'] == 5) & df['is_omission'].isna()
    rxrr = (df['task_block_number'] == 5) & (df['correct'] == 1) & (df['is_omission'] == 1) & (df['stimulus_number'] == 3)
    rrxr = (df['task_block_number'] == 5) & (df['correct'] == 1) & (df['is_omission'] == 1) & (df['stimulus_number'] == 4)
    rrrx = (df['task_block_number'] == 5) & (df['correct'] == 1) & (df['is_omission'] == 1) & (df['stimulus_number'] >= 5)

    return {
        'AAAB': aaab, 'AXAB': axab, 'AAXB': aaxb, 'AAAX': aaax,
        'BBBA': bbba, 'BXBA': bxba, 'BBXA': bbxa, 'BBBX': bbbx,
        'RRRR': rrrr, 'RXRR': rxrr, 'RRXR': rrxr, 'RRRX': rrrx
    }


def extract_good_units(nwb):
    """
    Extracts 'good' quality units from an NWB file and maps them to probes.
    
    Args:
        nwb: An opened pynwb.NWBFile object.
        
    Returns:
        pd.DataFrame: A DataFrame of units with quality '1' or 'good', mapped to 'probeA', 'probeB', or 'probeC'.
    """
    if nwb.units is None:
        return pd.DataFrame()
        
    df_units = nwb.units.to_dataframe()
    
    # 1. Filter Good Units (quality == 1 or 'good')
    if 'quality' in df_units.columns:
        mask_good = df_units['quality'].astype(str).isin(['1', '1.0', 'good', 'good_unit'])
    else:
        # Fallback to presence_ratio if quality is missing
        mask_good = (pd.to_numeric(df_units['presence_ratio'], errors='coerce') > 0.9) if 'presence_ratio' in df_units.columns else (df_units.index == df_units.index)
        
    good_units = df_units[mask_good].copy()
    
    # 2. Map to Probes based on peak_channel_id
    # Probe A (0-127), Probe B (128-255), Probe C (256-383)
    def map_probe(channel_id):
        if pd.isna(channel_id): return 'unknown'
        cid = int(channel_id)
        if 0 <= cid < 128: return 'probeA'
        elif 128 <= cid < 256: return 'probeB'
        elif 256 <= cid < 384: return 'probeC'
        return 'unknown'
        
    if 'peak_channel_id' in good_units.columns:
        good_units['probe'] = good_units['peak_channel_id'].apply(map_probe)
    elif 'peak_channel' in good_units.columns:
        good_units['probe'] = good_units['peak_channel'].apply(map_probe)
        
    return good_units


def epoch_timeseries_data(timeseries_data, fs, trial_start_times, t_pre_s=1.0, t_post_s=4.0):
    """
    Extracts fixed-length epochs from continuous timeseries data (e.g. LFP) aligned to trial events.
    
    Args:
        timeseries_data (np.ndarray): The continuous signal data of shape (n_samples, n_channels).
        fs (float): Sampling frequency in Hz.
        trial_start_times (list or np.ndarray): Array of timestamps (in seconds) to align to.
        t_pre_s (float): Seconds to extract before the trial_start_time.
        t_post_s (float): Seconds to extract after the trial_start_time.
        
    Returns:
        np.ndarray: Epoched data of shape (n_trials, n_channels, n_samples_per_epoch).
    """
    n_trials = len(trial_start_times)
    n_channels = timeseries_data.shape[1] if len(timeseries_data.shape) > 1 else 1
    n_samples_per_epoch = int((t_pre_s + t_post_s) * fs)
    
    epoched_data = np.zeros((n_trials, n_channels, n_samples_per_epoch))
    
    for i, t_start in enumerate(trial_start_times):
        start_idx = int((t_start - t_pre_s) * fs)
        end_idx = start_idx + n_samples_per_epoch
        
        if start_idx >= 0 and end_idx <= timeseries_data.shape[0]:
            if n_channels == 1:
                # Transpose 1D
                epoched_data[i, 0, :] = timeseries_data[start_idx:end_idx]
            else:
                # Transpose to match [trials, channels, time]
                epoched_data[i, :, :] = timeseries_data[start_idx:end_idx, :].T
            
    return epoched_data
