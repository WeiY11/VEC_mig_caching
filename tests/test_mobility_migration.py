
import sys
import os
import logging
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.system_simulator import CompleteSystemSimulator
from config import config

def verify_improvements():
    logging.basicConfig(level=logging.INFO)
    print("🚀 Starting verification of improvements...")

    # Initialize simulator
    sim = CompleteSystemSimulator()
    print("✅ Simulator initialized.")

    # Enable central resource to trigger _update_central_scheduler
    sim._central_resource_enabled = True
    
    print("📍 RSU Positions:")
    for rsu in sim.rsus:
        print(f"  {rsu['id']}: {rsu['position']}")
    
    # Mock some vehicles and RSUs to trigger migration
    # Place a vehicle near the edge of RSU_0's coverage
    rsu0 = sim.rsus[0]
    rsu0['coverage_radius'] = 400.0 # Reduce coverage to ensure we can be at edge but still closest
    rsu0_pos = rsu0['position']
    
    # Move other RSUs far away to ensure RSU_0 is the closest
    sim.rsus[1]['position'] = np.array([2000.0, 500.0])
    sim.rsus[2]['position'] = np.array([3000.0, 500.0])
    sim.rsus[3]['position'] = np.array([4000.0, 500.0])
    
    coverage = rsu0['coverage_radius']
    
    # Vehicle at 95% of coverage radius, moving away
    if len(rsu0_pos) == 2:
        offset = np.array([380.0, 0])
    else:
        offset = np.array([380.0, 0, 0])
        
    vehicle_pos = rsu0_pos + offset
    
    # Update vehicle 0
    sim.vehicles[0]['position'] = vehicle_pos
    sim.vehicles[0]['direction'] = 0.0 # Moving East (away from RSU if RSU is at 0,0)
    
    # Add a task to RSU_0 queue belonging to vehicle 0
    task = {
        'id': 'test_task_1',
        'vehicle_id': sim.vehicles[0]['id'],
        'data_size': 1.0,
        'compute_cycles': 1e9,
        'deadline': 1.0,
        'timestamp': sim.current_time
    }
    rsu0['computation_queue'].append(task)
    
    print(f"🚗 Vehicle 0 placed at {vehicle_pos}, RSU 0 at {rsu0_pos}, Coverage: {coverage}")
    print(f"📋 RSU 0 Queue before check: {len(rsu0['computation_queue'])}")

    # Run check directly
    print("🔄 Checking mobility migration directly...")
    mobility_migrations = sim._check_mobility_migration()
    
    # Check if migration happened
    rsu0_queue_len = len(rsu0['computation_queue'])
    print(f"📋 RSU 0 Queue after check: {rsu0_queue_len}")
    
    print(f"📊 Mobility Migrations Triggered: {mobility_migrations}")
    sys.stdout.flush()
    
    if mobility_migrations > 0:
        print("✅ SUCCESS: Mobility migration triggered successfully!")
    else:
        print("⚠️ WARNING: No mobility migration triggered. Check conditions.")
        # Debug info
        print(f"Vehicle 0 pos: {sim.vehicles[0]['position']}")
        dist = np.linalg.norm(sim.vehicles[0]['position'] - rsu0['position'])
        print(f"Distance to RSU 0: {dist}")
        print(f"Threshold: {coverage * 0.9}")
        print(f"Vehicle direction: {sim.vehicles[0]['direction']}")
    sys.stdout.flush()

if __name__ == "__main__":
    verify_improvements()
