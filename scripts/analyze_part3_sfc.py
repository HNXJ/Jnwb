import h5py
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import os
import sys

# Analysis Configuration
TARGET_BAND = (38, 40)
FS = 1000  # 1kHz
WINDOW = (3.0, 4.5) # Seconds post-fixation (Actual Omission period)

def filter_lfp(data, low, high, fs):
    nyq = 0.5 * fs
    b = signal.firwin(101, [low/nyq, high/nyq], pass_zero=False)
    return signal.filtfilt(b, [1.0], data, axis=0)

def rayleigh_p_value(r, n):
    if n == 0: return 1.0
    z = n * (r**2)
    return np.exp(-z)

def calculate_plv(phases):
    if len(phases) == 0: return 0, 0, 1.0
    vectors = np.exp(1j * phases)
    mean_vector = np.mean(vectors)
    plv = np.abs(mean_vector)
    angle = np.angle(mean_vector)
    p_val = rayleigh_p_value(plv, len(phases))
    return plv, angle, p_val

def main(session_id):
    input_h5 = f'D:/hnxj-gemini/ses-{session_id}_data_chunks.h5'
    summary_path = f'D:/hnxj-gemini/ses-{session_id}_part1_summary.csv'
    
    if not os.path.exists(input_h5) or not os.path.exists(summary_path):
        print(f"Missing files for {session_id}.")
        return

    # Load top units from summary
    df_summary = pd.read_csv(summary_path)
    target_units = df_summary.sort_values(by='omit_index', ascending=False).head(10)['unit_idx'].astype(str).tolist()
    
    results = {}
    with h5py.File(input_h5, 'r') as f:
        target_modes = ['mode_1', 'mode_5']
        
        for u_id in target_units:
            results[u_id] = {}
            plt.figure(figsize=(10, 5))
            
            for i, mode_str in enumerate(target_modes):
                if mode_str not in f: continue
                
                mode_grp = f[mode_str]
                all_phases = []
                
                for t_key in mode_grp.keys():
                    trial_grp = mode_grp[t_key]
                    lfp = trial_grp['lfp/pfc'][()] if 'lfp/pfc' in trial_grp else trial_grp['lfp'][()][:, 0]
                    if len(lfp.shape) > 1: lfp = lfp[:, 0]
                    
                    lfp_filt = filter_lfp(lfp, TARGET_BAND[0], TARGET_BAND[1], FS)
                    phase_ts = np.angle(signal.hilbert(lfp_filt))
                    
                    if 'spikes' in trial_grp and u_id in trial_grp['spikes']:
                        spike_times = trial_grp['spikes'][u_id][()]
                        mask = (spike_times >= WINDOW[0]) & (spike_times <= WINDOW[1])
                        for ts in spike_times[mask]:
                            idx = int(1000 + ts * FS)
                            if 0 <= idx < 6000: all_phases.append(phase_ts[idx])

                plv, mean_angle, p_val = calculate_plv(np.array(all_phases))
                results[u_id][mode_str] = {'plv': plv, 'p_val': p_val, 'n_spikes': len(all_phases)}
                
                ax = plt.subplot(1, 2, i+1, projection='polar')
                counts, bins = np.histogram(all_phases, bins=30, range=(-np.pi, np.pi))
                ax.bar(bins[:-1], counts, width=2*np.pi/30, color='gold' if mode_str=='mode_5' else 'gray', alpha=0.7)
                ax.set_title(f"{mode_str}\nPLV={plv:.3f}, p={p_val:.4f}")

            plt.suptitle(f"Unit {u_id} PLV to 38-40Hz Gamma ({session_id})")
            plt.savefig(f"D:/figures/ses-{session_id}_unit_{u_id}_plv.png")
            plt.close()

if __name__ == "__main__":
    import pandas as pd
    if len(sys.argv) < 2:
        print("Usage: python analyze_part3_sfc.py <session_id>")
    else:
        main(sys.argv[1])
