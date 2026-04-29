import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

class AlphaAblationEnv(gym.Env):
    def __init__(self, df, factor_columns=None, label_column='target'):
        super().__init__()
        self.df = df.reset_index(drop=True)
        
        # If no factors specified, use all columns except the label
        self.factors = factor_columns if factor_columns else [c for c in df.columns if c != label_column]
        self.label = label_column
        
        # Action: 0=Neutral, 1=Long, 2=Short
        self.action_space = spaces.Discrete(3)
        
        # Observation: The current row of alpha factors
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(len(self.factors),), 
            dtype=np.float32
        )
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        return self._get_obs(), {}

    def step(self, action):
        # Reward logic: action * next_period_return
        target_return = self.df.loc[self.current_step, self.label]
        
        reward = 0.0
        if action == 1: reward = target_return     # Long
        elif action == 2: reward = -target_return  # Short
        
        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        
        return self._get_obs(), reward, terminated, False, {}

    def _get_obs(self):
        return self.df.loc[self.current_step, self.factors].values.astype(np.float32)