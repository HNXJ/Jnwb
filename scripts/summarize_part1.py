import os
import json
import pandas as pd
import numpy as np

def aggregate_session_results(checkpoint_dir, output_path):
    """Aggregates per-unit results into a single session summary."""
    all_unit_data = []
    
    files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.json')]
    print(f"Aggregating {len(files)} unit checkpoints...")
    
    for filename in files:
        with open(os.path.join(checkpoint_dir, filename), 'r') as f:
            data = json.load(f)
            unit_idx = data['unit_idx']
            
            # Flatten the analysis dictionary for easier DataFrame creation
            for mode_id, analysis in data['analysis'].items():
                row = {
                    'unit_idx': unit_idx,
                    'mode_id': int(mode_id),
                    'mode_label': analysis['label'],
                    'n_trials': analysis['n_trials']
                }
                # Add epoch means
                for epoch, mean_val in analysis['epochs'].items():
                    row[f'epoch_{epoch}'] = mean_val
                
                all_unit_data.append(row)
    
    df = pd.DataFrame(all_unit_data)
    df.to_csv(output_path, index=False)
    print(f"Aggregated summary saved to {output_path}")
    
    # Simple Ranking: Units with highest Omission response in Mode 5 (AAAx)
    # Relative to baseline
    if not df.empty and 'epoch_omission' in df.columns and 'epoch_baseline' in df.columns:
        mode_5 = df[df['mode_id'] == 5].copy()
        mode_5['omission_index'] = mode_5['epoch_omission'] / (mode_5['epoch_baseline'] + 1e-6)
        top_omission_units = mode_5.sort_values(by='omission_index', ascending=False).head(10)
        print("\nTop 10 Omission-Responding Units (Mode 5 - AAAx):")
        print(top_omission_units[['unit_idx', 'epoch_baseline', 'epoch_omission', 'omission_index']])

if __name__ == "__main__":
    aggregate_session_results(
        checkpoint_dir=r'D:\hnxj-gemini\checkpoints\ses-230818',
        output_path=r'D:\hnxj-gemini\ses-230818_part1_summary.csv'
    )
