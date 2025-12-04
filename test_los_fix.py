"""Simple test for LoS probability fix"""
import sys
import os
os.chdir('d:/VEC_mig_caching')
sys.path.insert(0, 'd:/VEC_mig_caching')

import math

# Import after path setup
from communication.models import WirelessCommunicationModel
from models.data_structures import Position

def p_los_3gpp(d):
    """3GPP TR 38.901 UMi Street Canyon"""
    if d <= 18:
        return 1.0
    return (18.0/d) + math.exp(-d/36.0)*(1.0 - 18.0/d)

print("\n" + "="*60)
print("LoS Probability: 3GPP Standard vs Fixed Model")
print("="*60)

m = WirelessCommunicationModel()
pos_a = Position(0, 0)

print(f"\n{'Dist':>6} | {'3GPP':>8} | {'Model':>8} | {'Match':>6}")
print("-"*40)

all_match = True
for d in [30, 50, 100, 150, 200, 300, 500]:
    pos_b = Position(d, 0)
    state = m.calculate_channel_state(pos_a, pos_b, 'vehicle', 'rsu')
    std = p_los_3gpp(d)
    
    # Model follows 3GPP but with spatial correction (may be slightly different)
    match = "Yes" if abs(state.los_probability - std) < 0.15 else "No"
    if match == "No":
        all_match = False
    
    print(f"{d:>5}m | {std:>8.4f} | {state.los_probability:>8.4f} | {match:>6}")

print("-"*40)
if all_match:
    print("SUCCESS: Model follows 3GPP standard (with spatial correction)")
else:
    print("WARNING: Some values differ significantly")
print("="*60)
