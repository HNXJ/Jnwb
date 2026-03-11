"""
Configuration for the Omission Neurons Project.
Includes trial condition mapping and epoch definitions.
"""

# Condition Mapping (task_condition_number)
# Based on the Visual Omission Oddball Paradigm
OMISSION_CONFIG = {
    1: {'sequence': 'AAAB', 'omission_idx': 4, 'label': 'Stim_A_Standard'},
    2: {'sequence': 'AAAB', 'omission_idx': 4, 'label': 'Stim_A_Control'},
    3: {'sequence': 'AxAB', 'omission_idx': 2, 'label': 'Omit_Pos2_A'},
    4: {'sequence': 'AAxB', 'omission_idx': 3, 'label': 'Omit_Pos3_A'},
    5: {'sequence': 'AAAx', 'omission_idx': 4, 'label': 'Omit_Pos4_A'},
    6: {'sequence': 'BBBA', 'omission_idx': 4, 'label': 'Stim_B_Standard'},
    7: {'sequence': 'BBBA', 'omission_idx': 4, 'label': 'Stim_B_Control'},
    8: {'sequence': 'BxBA', 'omission_idx': 2, 'label': 'Omit_Pos2_B'},
    9: {'sequence': 'BBxA', 'omission_idx': 3, 'label': 'Omit_Pos3_B'},
    10: {'sequence': 'BBBx', 'omission_idx': 4, 'label': 'Omit_Pos4_B'},
    # Modes 11 and 12 are typically control full sequences
    11: {'sequence': 'AAAA', 'omission_idx': None, 'label': 'Control_AAAA'},
    12: {'sequence': 'BBBB', 'omission_idx': None, 'label': 'Control_BBBB'}
}

# Epoch definitions relative to the omission/event time (t=0) in seconds
EPOCHS = {
    'fixation': (-1.0, -0.5),
    'baseline': (-0.5, -0.1),
    'stimulus_on': (-0.1, 0.1),
    'omission': (0.0, 0.5),
    'delay': (0.5, 2.0)
}

# Sampling rates (to be verified from data)
FS_LFP = 1000  # Default LFP sampling rate in Hz
FS_SPIKE = 30000 # Default spike sampling rate in Hz
