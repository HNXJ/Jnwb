import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# --- Config ---
INPUT_H5 = r'D:\OmissionAnalysis\spikes_by_condition_ses-230818.h5'
TARGET_CONDITION = 'AAAB'
OUTPUT_PNG = r'D:\OmissionAnalysis\grand_timing_validation.png'

# Event timings relative to Stimulus 1 onset (t=0) in seconds
EVENT_TIMESTAMPS = {
    "Fixation": -0.5,
    "Stim 1": 0.0,
    "Stim 2": 1.0,
    "Stim 3": 2.0,
    "Stim 4": 3.0
}

def main():
    with h5py.File(INPUT_H5, 'r') as f:
        if TARGET_CONDITION not in f:
            print(f"Condition '{TARGET_CONDITION}' not found.")
            return
            
        # Load spiking data [trials, units, time]
        spikes = f[f'{TARGET_CONDITION}/spiking_activity'][()]
        
        # 1. Average across trials and units to get Population Average [time]
        pop_avg = np.mean(np.mean(spikes, axis=0), axis=0)
        
        # 2. Smooth with 50ms Gaussian kernel
        smoothed_avg = gaussian_filter1d(pop_avg * 1000, sigma=50) # Convert to Hz
        
        # 3. Create Time Axis (-1s to +5s relative to Stim 1)
        time_axis = np.linspace(-1000, 5000, 6000)

        # 4. Plot
        plt.figure(figsize=(15, 7), dpi=150)
        plt.plot(time_axis, smoothed_avg, color='#CFB87C', linewidth=2, label=f'Population Average ({TARGET_CONDITION})')
        
        # Add event markers
        colors = ['cyan', 'white', 'white', 'white', 'gold']
        for i, (event, ts) in enumerate(EVENT_TIMESTAMPS.items()):
            plt.axvline(ts * 1000, color=colors[i], linestyle='--', alpha=0.7, label=event)
            
        plt.xlabel("Time from Stimulus 1 Onset (ms)")
        plt.ylabel("Firing Rate (Hz)")
        plt.title(f"Grand Timing Validation: All 4 Stimuli (Session 230818, N={spikes.shape[1]} units)")
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.1)
        plt.xlim(-1000, 5000)
        
        # Add a secondary axis for Eye/Pupil if we had them epoched here (TODO)
        
        plt.savefig(OUTPUT_PNG)
        print(f"Grand validation plot saved to {OUTPUT_PNG}")

if __name__ == "__main__":
    main()
