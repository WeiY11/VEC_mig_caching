import multiprocessing as mp
import numpy as np
import time
from typing import List, Dict, Any, Optional, Union, Tuple
import logging

# Worker function to run in a separate process
def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    try:
        env = env_fn_wrapper.x()
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                next_state, reward, done, info = env.step(data)
                if done:
                    # Auto-reset on done is common in vectorized envs
                    # But for our specific training loop, we might want to control reset explicitly
                    # or return the reset state immediately. 
                    # Standard Gym VectorEnv auto-resets. Let's follow that pattern.
                    # However, our training loop might expect 'done' to be True to handle episode end logic.
                    # Let's return done=True and let the main process handle reset if needed, 
                    # OR we can return the reset state in 'info' like Gym.
                    
                    # For simplicity and compatibility with existing loop:
                    # We will NOT auto-reset here. The main process sees done=True, 
                    # records stats, and then calls reset() explicitly.
                    pass
                remote.send((next_state, reward, done, info))
            elif cmd == 'reset':
                state = env.reset_environment()
                remote.send(state)
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_attr':
                attr_name = data
                if hasattr(env, attr_name):
                    remote.send(getattr(env, attr_name))
                else:
                    remote.send(None)
            elif cmd == 'set_attr':
                attr_name, value = data
                setattr(env, attr_name, value)
                remote.send(None)
            else:
                raise NotImplementedError(f"Unknown command: {cmd}")
    except Exception as e:
        logging.error(f"Worker process failed: {e}")
        remote.close()

class CloudpickleWrapper:
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to pickle
    functions which can fail for lambdas or local functions).
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

class VectorizedSingleAgentEnvironment:
    """
    Vectorized environment that runs multiple SingleAgentTrainingEnvironment instances in parallel.
    """
    def __init__(self, env_fns: List[callable]):
        import traceback
        print("DEBUG: VectorizedSingleAgentEnvironment instantiated!")
        traceback.print_stack()
        self.num_envs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(self.num_envs)])
        self.ps = [
            mp.Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)
        ]
        for p in self.ps:
            p.daemon = True  # clean up if main process dies
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def reset(self) -> np.ndarray:
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        return np.stack(results)

    def step(self, actions: Union[np.ndarray, List[Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        # actions: (num_envs, action_dim) or list of actions
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def get_attr(self, attr_name: str) -> List[Any]:
        for remote in self.remotes:
            remote.send(('get_attr', attr_name))
        return [remote.recv() for remote in self.remotes]

    def set_attr(self, attr_name: str, values: List[Any]):
        # values should be a list of length num_envs
        for remote, value in zip(self.remotes, values):
            remote.send(('set_attr', (attr_name, value)))
            remote.recv() # Wait for ack

    @property
    def num_vehicles(self):
        # Assume all envs have same config
        self.remotes[0].send(('get_attr', 'num_vehicles'))
        return self.remotes[0].recv()

    @property
    def num_rsus(self):
        self.remotes[0].send(('get_attr', 'num_rsus'))
        return self.remotes[0].recv()

    @property
    def num_uavs(self):
        self.remotes[0].send(('get_attr', 'num_uavs'))
        return self.remotes[0].recv()
