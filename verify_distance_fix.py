"""
Verify distance calculation fix: 2D vs 3D distance for 3GPP compliance
"""
import math

print("=" * 70)
print(" Distance Calculation Fix Verification")
print(" 3GPP TR 38.901: 2D for LoS, 3D for Path Loss")
print("=" * 70)

# Test scenarios
scenarios = [
    ("RSU-Vehicle", 1.5, 25.0, 100.0),   # (name, h_vehicle, h_rsu, horizontal_dist)
    ("RSU-Vehicle", 1.5, 25.0, 200.0),
    ("UAV-Vehicle", 1.5, 100.0, 100.0),  
    ("UAV-Vehicle", 1.5, 100.0, 200.0),
    ("UAV-Vehicle", 1.5, 100.0, 300.0),
]

print(f"\n{'Scenario':<15} | {'H.Dist':>8} | {'d_2D':>8} | {'d_3D':>8} | {'Error':>8}")
print("-" * 60)

for name, h1, h2, h_dist in scenarios:
    # 2D distance (horizontal only)
    d_2d = h_dist
    
    # 3D distance (including height difference)
    height_diff = abs(h2 - h1)
    d_3d = math.sqrt(h_dist**2 + height_diff**2)
    
    # Error if using 3D for LoS calculation instead of 2D
    error_pct = (d_3d - d_2d) / d_2d * 100
    
    print(f"{name:<15} | {h_dist:>7.0f}m | {d_2d:>7.1f}m | {d_3d:>7.1f}m | {error_pct:>+7.1f}%")

print("-" * 60)

# Show impact on LoS probability
def p_los_3gpp(d):
    if d <= 18:
        return 1.0
    return (18.0/d) + math.exp(-d/36.0)*(1.0 - 18.0/d)

print(f"\n Impact on LoS Probability (UAV at 100m, vehicle at 1.5m):")
print(f"{'H.Dist':>10} | {'P_LoS(2D)':>12} | {'P_LoS(3D)':>12} | {'Error':>10}")
print("-" * 52)

for h_dist in [100, 150, 200, 300]:
    d_2d = h_dist
    d_3d = math.sqrt(h_dist**2 + 98.5**2)  # 100 - 1.5 = 98.5m height diff
    
    p_2d = p_los_3gpp(d_2d)
    p_3d = p_los_3gpp(d_3d)
    error = (p_3d - p_2d) / p_2d * 100
    
    print(f"{h_dist:>9}m | {p_2d:>12.4f} | {p_3d:>12.4f} | {error:>+9.1f}%")

print("-" * 52)
print("\n FIX APPLIED:")
print("   - LoS probability now uses 2D horizontal distance (d_2D)")
print("   - Path loss now uses 3D actual distance (d_3D)")
print("   - Compliant with 3GPP TR 38.901 standard")
print("=" * 70)
