import h5py
import numpy as np
import mne
import os
import matplotlib.pyplot as plt
import sys
from config import OMISSION_CONFIG, EPOCHS

# Analysis Configuration
FS = 1000  # 1kHz sampling rate
FREQS = np.arange(10, 100, 2)  # 10Hz to 100Hz in 2Hz steps
N_CYCLES = FREQS / 2.0  # Adaptive cycle count for Morlet Wavelets

def compute_trial_tfr(lfp_data):
    # Reshape for MNE: [epochs, channels, times]
    # We'll pick a subset of channels (e.g., first 32) to save time/RAM
    data = lfp_data.T[np.newaxis, :32, :] # (1, 32, 6000)
    
    # Compute Power using Morlet Wavelets
    power = mne.time_frequency.tfr_array_morlet(
        data, sfreq=FS, freqs=FREQS, n_cycles=N_CYCLES, 
        output='power', n_jobs=1, verbose=False
    )
    # Clear MNE cache if possible
    mne.utils.set_config('MNE_USE_NUMBA', 'false') # Numba can cause segfaults in tight loops
    return np.mean(power[0], axis=0)

def main(session_id):
    input_h5 = f'D:/hnxj-gemini/ses-{session_id}_data_chunks.h5'
    output_tfr = f'D:/hnxj-gemini/ses-{session_id}_tfr_results.h5'
    
    if not os.path.exists(input_h5):
        print(f"Input file {input_h5} not found.")
        return

    with h5py.File(input_h5, 'r') as f_in, h5py.File(output_tfr, 'a') as f_out:
        target_modes = ['mode_1', 'mode_5']
        
        for mode_str in target_modes:
            if mode_str not in f_in:
                continue
                
            print(f"Processing TFR for {session_id} {mode_str}...")
            mode_grp = f_in[mode_str]
            trial_keys = list(mode_grp.keys())
            
            all_trial_power = []
            
            for i, t_key in enumerate(trial_keys):
                lfp = mode_grp[t_key]['lfp'][()] # Load (6000, 128)
                trial_power = compute_trial_tfr(lfp)
                all_trial_power.append(trial_power)
                
                if (i + 1) % 50 == 0:
                    print(f"  {session_id} {mode_str}: Processed {i+1}/{len(trial_keys)} trials...")

            mean_power = np.mean(all_trial_power, axis=0)
            baseline = mean_power[:, 500:900].mean(axis=1, keepdims=True)
            db_power = 10 * np.log10(mean_power / baseline)
            
            if mode_str in f_out:
                del f_out[mode_str]
            f_out.create_dataset(f"{mode_str}/db_power", data=db_power, compression='gzip')

            # Visualization
            time_axis = np.linspace(-1000, 5000, 6000)
            plt.figure(figsize=(12, 6))
            plt.pcolormesh(time_axis, FREQS, db_power, shading='auto', cmap='RdBu_r', vmin=-3, vmax=3)
            plt.colorbar(label='Power (dB)')
            plt.title(f"PFC TFR - {session_id} {mode_str}")
            plt.savefig(f"D:/figures/ses-{session_id}_{mode_str}_tfr.png")
            plt.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_part2_tfr.py <session_id>")
    else:
        main(sys.argv[1])
