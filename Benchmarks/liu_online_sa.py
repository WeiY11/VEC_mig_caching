#!/usr/bin/env python3
"""
Online Simulated Annealing baseline (Liu & Cao 2022 style).

Use this to adapt offloading/cache parameters on the fly without full RL retraining.
Works with a callable `evaluate_fn(params) -> reward` that returns higher-is-better.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence

import numpy as np


@dataclass
class SAConfig:
    init_temp: float = 1.0
    min_temp: float = 0.001
    cooling: float = 0.9
    max_iters: int = 800
    step_scale: float = 0.1
    seed: int = 42


class OnlineSimulatedAnnealing:
    def __init__(self, dim: int, bounds: Sequence[tuple[float, float]], cfg: SAConfig):
        assert len(bounds) == dim, "bounds length must match dim"
        self.dim = dim
        self.bounds = np.array(bounds, dtype=np.float32)
        self.cfg = cfg
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        self.current = self._init_point()
        self.current_score = None

    def _init_point(self) -> np.ndarray:
        low, high = self.bounds[:, 0], self.bounds[:, 1]
        return low + np.random.rand(self.dim) * (high - low)

    def _perturb(self, x: np.ndarray) -> np.ndarray:
        low, high = self.bounds[:, 0], self.bounds[:, 1]
        step = (np.random.randn(self.dim) * self.cfg.step_scale * (high - low)).astype(np.float32)
        cand = np.clip(x + step, low, high)
        return cand

    def search(self, evaluate_fn: Callable[[np.ndarray], float]) -> dict:
        temp = self.cfg.init_temp
        best = self.current
        best_score = evaluate_fn(best)
        self.current_score = best_score

        history: List[float] = [best_score]

        for i in range(self.cfg.max_iters):
            cand = self._perturb(self.current)
            cand_score = evaluate_fn(cand)
            delta = cand_score - self.current_score
            accept = delta > 0 or math.exp(delta / max(temp, 1e-6)) > random.random()
            if accept:
                self.current, self.current_score = cand, cand_score
                if cand_score > best_score:
                    best, best_score = cand, cand_score
            history.append(best_score)
            temp = max(self.cfg.min_temp, temp * self.cfg.cooling)
        return {"best_params": best, "best_score": best_score, "history": history}

    def select_action(self, state: np.ndarray = None) -> np.ndarray:
        """
        Online action selection interface for RL environment compatibility.
        
        This returns the current best parameters as an action vector.
        For true online adaptation, call `step()` periodically with reward feedback.
        
        Args:
            state: Current environment state (not used in basic SA, but kept for interface compatibility)
        
        Returns:
            Current action/parameter vector
        """
        return self.current.copy()
    
    def step(self, reward: float) -> np.ndarray:
        """
        Online adaptation step: perturb current solution and decide whether to accept.
        
        This enables true online SA where each call adapts based on the latest reward.
        
        Args:
            reward: Reward from the last action (higher is better)
            
        Returns:
            New action vector after one SA step
        """
        if self.current_score is None:
            self.current_score = reward
            return self.current.copy()
        
        # Generate candidate
        cand = self._perturb(self.current)
        
        # Acceptance decision (using reward as score)
        delta = reward - self.current_score
        temp = getattr(self, '_online_temp', self.cfg.init_temp)
        accept = delta > 0 or math.exp(delta / max(temp, 1e-6)) > random.random()
        
        if accept:
            self.current = cand
            self.current_score = reward
        
        # Cool down temperature
        self._online_temp = max(self.cfg.min_temp, temp * self.cfg.cooling)
        
        return self.current.copy()
    
    def select_action_with_dim(self, state: np.ndarray, action_dim: int) -> np.ndarray:
        """Wrapper to ensure action dimension matches environment."""
        action = self.select_action(state)
        if len(action) < action_dim:
            padding = np.zeros(action_dim - len(action), dtype=np.float32)
            action = np.concatenate([action, padding])
        elif len(action) > action_dim:
            action = action[:action_dim]
        return action


__all__ = ["SAConfig", "OnlineSimulatedAnnealing"]

