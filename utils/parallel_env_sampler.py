#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 并行环境采样器 - 提高CPU利用率

通过多进程并行运行多个环境实例，加速数据采集。
适用于强化学习训练中CPU利用率低的情况。

使用方法：
    from utils.parallel_env_sampler import ParallelEnvSampler
    
    sampler = ParallelEnvSampler(
        env_fn=create_env_fn,
        num_envs=4,  # 并行环境数
    )
    
    # 并行采样
    experiences = sampler.sample(agent, num_steps=100)

作者：VEC_mig_caching Team
"""

import os
import numpy as np
from typing import Callable, List, Dict, Any, Optional, Tuple
from multiprocessing import Process, Pipe, Queue
import queue
import threading
import time


class EnvWorker(Process):
    """环境工作进程"""
    
    def __init__(
        self,
        env_fn: Callable,
        conn,
        worker_id: int,
    ):
        super().__init__()
        self.env_fn = env_fn
        self.conn = conn
        self.worker_id = worker_id
        
    def run(self):
        """工作进程主循环"""
        env = self.env_fn()
        
        while True:
            try:
                cmd, data = self.conn.recv()
                
                if cmd == 'step':
                    action = data
                    next_state, reward, done, info = env.step(action)
                    if done:
                        next_state = env.reset()
                    self.conn.send(('step_result', (next_state, reward, done, info)))
                    
                elif cmd == 'reset':
                    state = env.reset()
                    self.conn.send(('reset_result', state))
                    
                elif cmd == 'get_state':
                    state = env.get_state()
                    self.conn.send(('state', state))
                    
                elif cmd == 'close':
                    break
                    
            except EOFError:
                break
                
        env.close() if hasattr(env, 'close') else None


class ParallelEnvSampler:
    """
    并行环境采样器
    
    通过多进程并行运行多个环境实例，加速数据采集。
    """
    
    def __init__(
        self,
        env_fn: Callable,
        num_envs: int = 4,
    ):
        """
        初始化并行采样器
        
        Args:
            env_fn: 创建环境的函数
            num_envs: 并行环境数量（建议设为CPU核心数的1/2到1倍）
        """
        self.env_fn = env_fn
        self.num_envs = num_envs
        
        self.workers = []
        self.parent_conns = []
        
        # 创建工作进程
        for i in range(num_envs):
            parent_conn, child_conn = Pipe()
            worker = EnvWorker(env_fn, child_conn, i)
            worker.start()
            self.workers.append(worker)
            self.parent_conns.append(parent_conn)
        
        # 重置所有环境获取初始状态
        self.states = self._reset_all()
        
        print(f"[ParallelEnvSampler] 已创建 {num_envs} 个并行环境")
    
    def _reset_all(self) -> List[np.ndarray]:
        """重置所有环境"""
        for conn in self.parent_conns:
            conn.send(('reset', None))
        
        states = []
        for conn in self.parent_conns:
            _, state = conn.recv()
            states.append(state)
        
        return states
    
    def step(self, actions: List[np.ndarray]) -> Tuple[List, List, List, List]:
        """
        并行执行一步
        
        Args:
            actions: 每个环境的动作列表
            
        Returns:
            next_states, rewards, dones, infos
        """
        # 发送动作
        for conn, action in zip(self.parent_conns, actions):
            conn.send(('step', action))
        
        # 接收结果
        next_states, rewards, dones, infos = [], [], [], []
        for i, conn in enumerate(self.parent_conns):
            _, (next_state, reward, done, info) = conn.recv()
            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
            self.states[i] = next_state
        
        return next_states, rewards, dones, infos
    
    def sample_batch(
        self,
        agent,
        num_steps: int = 100,
        training: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        并行采样一批经验
        
        Args:
            agent: 智能体（需要有select_action方法）
            num_steps: 采样步数
            training: 是否训练模式
            
        Returns:
            经验列表 [{'state', 'action', 'reward', 'next_state', 'done'}, ...]
        """
        experiences = []
        
        for _ in range(num_steps):
            # 并行获取动作
            actions = []
            for state in self.states:
                action = agent.select_action(state, training=training)
                actions.append(action)
            
            # 并行执行步骤
            current_states = [s.copy() for s in self.states]
            next_states, rewards, dones, infos = self.step(actions)
            
            # 收集经验
            for i in range(self.num_envs):
                experiences.append({
                    'state': current_states[i],
                    'action': actions[i],
                    'reward': rewards[i],
                    'next_state': next_states[i],
                    'done': dones[i],
                    'info': infos[i],
                })
        
        return experiences
    
    def close(self):
        """关闭所有工作进程"""
        for conn in self.parent_conns:
            conn.send(('close', None))
        
        for worker in self.workers:
            worker.join(timeout=1)
            if worker.is_alive():
                worker.terminate()
        
        print("[ParallelEnvSampler] 所有环境已关闭")


class AsyncExperienceBuffer:
    """
    异步经验缓冲区
    
    使用后台线程预取下一批经验，减少等待时间。
    """
    
    def __init__(
        self,
        sampler: ParallelEnvSampler,
        agent,
        buffer_size: int = 2,
        steps_per_batch: int = 50,
    ):
        """
        初始化异步缓冲区
        
        Args:
            sampler: 并行采样器
            agent: 智能体
            buffer_size: 预取缓冲区大小
            steps_per_batch: 每批采样步数
        """
        self.sampler = sampler
        self.agent = agent
        self.buffer_size = buffer_size
        self.steps_per_batch = steps_per_batch
        
        self.buffer = Queue(maxsize=buffer_size)
        self.running = True
        
        # 启动后台采样线程
        self.sample_thread = threading.Thread(target=self._sample_loop, daemon=True)
        self.sample_thread.start()
    
    def _sample_loop(self):
        """后台采样循环"""
        while self.running:
            try:
                experiences = self.sampler.sample_batch(
                    self.agent,
                    num_steps=self.steps_per_batch,
                    training=True,
                )
                self.buffer.put(experiences, timeout=1)
            except queue.Full:
                continue
            except Exception as e:
                print(f"[AsyncBuffer] 采样错误: {e}")
                break
    
    def get_batch(self, timeout: float = 5.0) -> Optional[List[Dict]]:
        """获取一批经验"""
        try:
            return self.buffer.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def stop(self):
        """停止后台采样"""
        self.running = False
        self.sample_thread.join(timeout=2)


def get_optimal_num_envs() -> int:
    """
    获取最优并行环境数量
    
    基于CPU核心数自动确定
    """
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    
    # 通常使用CPU核心数的1/2到3/4
    optimal = max(2, cpu_count // 2)
    
    # 限制最大值避免过多进程
    return min(optimal, 8)


def setup_gpu_optimization():
    """
    设置GPU优化环境变量
    
    调用此函数可以优化PyTorch的GPU性能
    """
    # 内存分配优化
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb=512'
    
    # 多线程优化
    os.environ['OMP_NUM_THREADS'] = str(max(1, os.cpu_count() // 2))
    os.environ['MKL_NUM_THREADS'] = str(max(1, os.cpu_count() // 2))
    
    # cuDNN优化
    try:
        import torch
        if torch.cuda.is_available():
            # 启用cuDNN自动调优
            torch.backends.cudnn.benchmark = False
            # 使用确定性算法（可选，可能降低性能）
            # torch.backends.cudnn.deterministic = True
            
            print(f"[GPU优化] cuDNN benchmark已启用")
            print(f"[GPU优化] GPU: {torch.cuda.get_device_name(0)}")
            print(f"[GPU优化] 可用显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    except ImportError:
        pass
    
    print(f"[CPU优化] OMP_NUM_THREADS = {os.environ.get('OMP_NUM_THREADS')}")
    print(f"[CPU优化] 推荐并行环境数: {get_optimal_num_envs()}")


if __name__ == "__main__":
    # 测试GPU优化设置
    setup_gpu_optimization()
