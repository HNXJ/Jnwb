"""
Trains a classifier to distinguish between Predictable and Random omissions.
"""
import sys
import os
import h5py
import numpy as np
import pandas as pd
from scipy.signal import welch

# Add jnwb to path
sys.path.append(r'D:\jnwb')
from jnwb.analysis import classify_omission_predictability

def get_power_features(spikes, fs, f_band=(38, 40)):
    """Calculates band power for each unit and trial."""
    n_trials, n_units, n_time = spikes.shape
    features = np.zeros((n_trials, n_units))
    
    for i in range(n_trials):
        for j in range(n_units):
            f, Pxx = welch(spikes[i, j, :], fs=fs, nperseg=256)
            band_mask = (f >= f_band[0]) & (f <= f_band[1])
            features[i, j] = np.mean(Pxx[band_mask])
            
    return features

def main(session_id):
    input_h5 = f'D:/OmissionAnalysis/spikes_by_condition_ses-{session_id}.h5'
    if not os.path.exists(input_h5):
        print(f"Data for {session_id} not found.")
        return

    with h5py.File(input_h5, 'r') as f:
        # 1. Prepare Predictable Data (AAAx)
        if 'AAAX' in f:
            predictable_spikes = f['AAAX/spiking_activity'][()]
            predictable_labels = np.ones(predictable_spikes.shape[0])
            predictable_features = get_power_features(predictable_spikes, fs=1000)
        else:
            print("AAAX condition not found.")
            return

        # 2. Prepare Random Data (RXRR + AXAB)
        random_features_list = []
        if 'RXRR' in f:
            random_spikes_rxrr = f['RXRR/spiking_activity'][()]
            random_features_list.append(get_power_features(random_spikes_rxrr, fs=1000))
        if 'AXAB' in f:
            random_spikes_axab = f['AXAB/spiking_activity'][()]
            random_features_list.append(get_power_features(random_spikes_axab, fs=1000))
        
        if not random_features_list:
            print("No Random omission conditions found.")
            return
            
        random_features = np.vstack(random_features_list)
        random_labels = np.zeros(random_features.shape[0])

        # 3. Combine and Run Classifier
        all_features = np.vstack([predictable_features, random_features])
        all_labels = np.concatenate([predictable_labels, random_labels])
        
        print(f"\n--- Classification Report for Session {session_id} ---")
        results = classify_omission_predictability(all_features, all_labels)
        
        print(f"  Mean Accuracy: {results['mean_accuracy']:.2f} +/- {results['std_accuracy']:.2f}")
        print(f"  (Predictable: {results['n_predictable']} trials, Random: {results['n_random']} trials)")
        
        # Get top 5 most important features (unit indices)
        top_units = np.argsort(results['feature_importances'])[::-1][:5]
        print(f"  Top 5 Discriminating Units: {top_units}")
        
if __name__ == "__main__":
    main("230818") # Test with session 230818
