import os
import subprocess
import pandas as pd
import json

# Configuration
PROJECT_DIR = r'D:\hnxj-gemini'
FIGURES_DIR = r'D:\figures'
NWB_ROOT = r'D:\CDOC\Analysis\reconstructed_nwbdata'

# Sessions to process (Targeting the primary biological targets first)
SESSIONS = [
    {
        'id': '230818',
        'nwb': os.path.join(NWB_ROOT, 'sub-C31o_ses-230818_rec.nwb'),
        'units': r'D:\oxm0818_units.npy'
    },
    {
        'id': '230825',
        'nwb': os.path.join(NWB_ROOT, 'sub-C31o_ses-230825_rec.nwb'),
        'units': r'D:\oxm0825_units.npy'
    }
]

def run_step(script_name, args):
    """Runs a pipeline step as a separate process to ensure RAM is cleared."""
    cmd = ['python', os.path.join(PROJECT_DIR, script_name)] + args
    print(f"\n>>> Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"!!! Error in {script_name}. Continuing to next step/session...")
    return result.returncode == 0

def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    for session in SESSIONS:
        sid = session['id']
        print(f"\n{'='*60}")
        print(f"STARTING FULL ANALYSIS FOR SESSION: {sid}")
        print(f"{'='*60}")
        
        # Step 1: Metadata Extraction
        # We'll use the existing extract_metadata.py but pass sid
        run_step('extract_metadata.py', [session['nwb'], sid])
        
        # Step 2: Part 1 - Find Omission Neurons
        # Identified units will be saved to a summary CSV
        run_step('analyze_part1.py', [sid, session['units']])
        
        # Step 3: Prepare Data Chunks (HDF5)
        # Using the session-specific metadata and units
        run_step('prepare_data_chunks.py', [sid, session['nwb']])
        
        # Step 4: Part 2 - LFP TFR
        run_step('analyze_part2_tfr.py', [sid])
        
        # Step 5: Part 3 - SFC (Spike-Phase Coupling)
        # This will read the top units from the summary CSV created in Step 2
        run_step('analyze_part3_sfc.py', [sid])
        
        # Step 6: Part 4 - Granger Causality
        run_step('analyze_part4_granger.py', [sid])
        
        print(f"\nFINISHED SESSION {sid}. RAM cleared. Moving to next...")

if __name__ == "__main__":
    main()
