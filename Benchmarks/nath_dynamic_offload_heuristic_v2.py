import numpy as np

class DynamicOffloadHeuristic:
    """
    A heuristic policy that:
    1. Parses the flat state vector into (Vehicles, RSUs, UAVs).
    2. For each vehicle, decides offloading based on:
       - Local queue length vs. RSU/UAV queue length
       - Channel quality (distance)
       - Energy levels
    3. Outputs a continuous action vector (or discrete, depending on usage).
       Here we output continuous actions compatible with the TD3 agent's action space.
    """
    def __init__(self, num_rsus: int, num_uavs: int):
        self.num_rsus = num_rsus
        self.num_uavs = num_uavs
        
        # Heuristic weights
        self.queue_weight = 0.6
        self.energy_weight = 0.2
        self.channel_weight = 0.2
        
        # Thresholds
        self.queue_threshold = 0.6  # If queue > 60%, try to offload
        self.dist_threshold = 0.8   # If distance > 80% (far), avoid
        
        # Weights for scoring
        self.delay_weight = 0.5
        self.energy_weight = 0.5

    def _score_node(self, vehicle, node, dist_norm):
        """Score a target node (RSU/UAV) for offloading. Higher is better."""
        # Simple utility: - (w1 * delay + w2 * energy)
        # Delay proxy: node queue length + transmission delay (dist)
        # Energy proxy: transmission energy (dist^2)
        
        weight_queue = 1.0
        weight_channel = 1.0
        
        # Channel quality is inverse of distance
        ch = 1.0 - dist_norm
        
        # Node load
        load_penalty = node.queue
        
        queue_penalty = weight_queue * node.queue
        return weight_channel * ch - (self.delay_weight * queue_penalty + self.energy_weight * load_penalty)

    def _parse_state(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Robustly split flat state: vehicles(?*5) + rsu(num_rsus*6) + uav(num_uavs*6) + global(8)."""
        flat = np.array(state, dtype=np.float32).reshape(-1)
        
        # Try different suffix lengths to find one that makes veh_len divisible by 5
        possible_suffixes = [8, 10, 24, 26] # 8 (global), 10 (global+?), 24 (global+central 16), 26 (global+central+?)
        
        veh_dim = 5
        rsu_dim = 6
        uav_dim = 6
        
        rsu_total = self.num_rsus * rsu_dim
        uav_total = self.num_uavs * uav_dim
        
        best_suffix = 8
        best_veh_len = -1
        
        for suffix in possible_suffixes:
            useful_len = len(flat) - suffix
            if useful_len <= 0:
                continue
            
            rem_len = useful_len - rsu_total - uav_total
            if rem_len > 0 and rem_len % veh_dim == 0:
                best_suffix = suffix
                best_veh_len = rem_len
                break
        
        if best_veh_len == -1:
            # Fallback to 8 if no match found, to let it fail naturally or handle it
            print(f"WARNING: Could not determine correct suffix for state len {len(flat)}. Defaulting to 8.")
            best_suffix = 8
            best_veh_len = len(flat) - 8 - rsu_total - uav_total

        print(f"DEBUG: flat.shape={flat.shape} suffix={best_suffix} veh_len={best_veh_len}")
        
        useful_len = len(flat) - best_suffix
        
        # Extract vehicle part
        veh_len = best_veh_len
        veh = flat[:veh_len]
        
        # Extract RSU part
        rsu_start = veh_len
        rsu_end = rsu_start + rsu_total
        rsu = flat[rsu_start:rsu_end]
        
        # Extract UAV part
        uav_start = rsu_end
        uav_end = uav_start + uav_total
        uav = flat[uav_start:uav_end]
        
        return (
            veh.reshape(-1, veh_dim) if veh_len > 0 else np.zeros((0, veh_dim), dtype=np.float32),
            rsu.reshape(-1, rsu_dim),
            uav.reshape(-1, uav_dim),
        )

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """
        Generate action vector.
        Action structure (OptimizedTD3):
        [
          offload_prob_local, offload_prob_rsu, offload_prob_uav,  (3)
          rsu_selection_probs (num_rsus),                          (4)
          uav_selection_probs (num_uavs),                          (2)
          control_params (10)                                      (10)
        ]
        Total dim = 3 + 4 + 2 + 10 = 19.
        """
        veh_states, rsu_states, uav_states = self._parse_state(state)
        
        # We need to generate ONE action vector for the whole system?
        # Wait, OptimizedTD3 agent generates one global action vector.
        # But this heuristic should probably generate actions per vehicle?
        # The environment expects a single action vector that represents the "Central Agent" decision?
        # OR does it expect per-vehicle actions?
        
        # OptimizedTD3Wrapper.decompose_action expects a single 1D array.
        # And it extracts:
        # offload_preference = base_segment[:3]
        # rsu_selection = ...
        # uav_selection = ...
        
        # This implies the action sets GLOBAL probabilities/preferences for ALL vehicles?
        # Yes, SingleAgentTrainingEnvironment applies these preferences to ALL vehicles 
        # (unless overridden by specific logic, but here it seems global).
        
        # So we need to calculate AVERAGE preference across all vehicles.
        
        local_scores = []
        rsu_scores = []
        uav_scores = []
        
        # Helper class for node state
        class NodeState:
            def __init__(self, s, is_veh=False):
                # s: [x, y, (z), cache, queue, energy]
                # Normalized values
                self.queue = s[3] if is_veh else s[3] # Index 3 is queue for all?
                # Vehicle: x, y, vel, queue, energy
                # RSU: x, y, cache, queue, energy, (cpu)
                # UAV: x, y, z, cache, energy, (cpu) -> Wait, UAV index 3 is cache?
                # Let's check indices in _parse_state or environment.
                # Vehicle: 0:x, 1:y, 2:vel, 3:queue, 4:energy
                # RSU: 0:x, 1:y, 2:cache, 3:queue, 4:energy, 5:cpu
                # UAV: 0:x, 1:y, 2:z, 3:cache, 4:energy, 5:cpu
                
                if is_veh:
                    self.queue = s[3]
                else:
                    # RSU/UAV queue is index 3?
                    # RSU: index 3 is queue. Correct.
                    # UAV: index 3 is cache? No.
                    # UAV: x, y, z, cache, energy.
                    # UAV doesn't have queue in state?
                    # OptimizedTD3Wrapper:
                    # uav_state = [x, y, z, cache, energy]
                    # It seems UAV queue is NOT in state?
                    # Or maybe index 4?
                    pass

        # Calculate average queue load
        avg_veh_queue = np.mean(veh_states[:, 3]) if len(veh_states) > 0 else 0
        
        # Decision logic
        # If load is high, prefer RSU/UAV
        # If load is low, prefer Local
        
        offload_prob_local = 1.0
        offload_prob_rsu = 0.0
        offload_prob_uav = 0.0
        
        if avg_veh_queue > self.queue_threshold:
            offload_prob_local = 0.2
            offload_prob_rsu = 0.5
            offload_prob_uav = 0.3
        
        # RSU selection: prefer RSU with lowest queue (index 3)
        # RSU state: x, y, cache, queue, energy, cpu
        rsu_queues = rsu_states[:, 3]
        # Softmax (negative queue)
        rsu_probs = np.exp(-5.0 * rsu_queues)
        rsu_probs /= (np.sum(rsu_probs) + 1e-6)
        
        # UAV selection: prefer UAV with lowest energy consumption (index 4)?
        # UAV state: x, y, z, cache, energy, cpu
        uav_energy = uav_states[:, 4]
        uav_probs = np.exp(-5.0 * uav_energy)
        uav_probs /= (np.sum(uav_probs) + 1e-6)
        
        # Construct action
        # 3 (offload) + num_rsus + num_uavs + 10 (control)
        
        action = np.concatenate([
            [offload_prob_local, offload_prob_rsu, offload_prob_uav],
            rsu_probs,
            uav_probs,
            np.zeros(10, dtype=np.float32) # Control params (ignored by heuristic usually)
        ])
        
        return action.astype(np.float32)

    def select_action_with_dim(self, state: np.ndarray, action_dim: int) -> np.ndarray:
        """Wrapper to ensure action dimension matches environment."""
        action = self.select_action(state)
        if len(action) < action_dim:
            # Pad with zeros
            padding = np.zeros(action_dim - len(action), dtype=np.float32)
            action = np.concatenate([action, padding])
        elif len(action) > action_dim:
            # Truncate
            action = action[:action_dim]
        return action
