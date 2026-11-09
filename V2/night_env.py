"""Optional training rewards; the original game and observations stay intact."""

import math
import gymnasium as gym
from gymnasium.utils import RecordConstructorArgs
from gym_env import WORLD_DIAGONAL


class GoalShaping(gym.Wrapper,RecordConstructorArgs):
    def __init__(self,env,scale=8.0,gamma=0.995):
        # Gymnasium can recreate this wrapper from its environment specification
        RecordConstructorArgs.__init__(self,scale=scale,gamma=gamma)
        gym.Wrapper.__init__(self,env)
        if not math.isfinite(scale) or scale<0:
            raise ValueError("scale must be finite and nonnegative")
        if not math.isfinite(gamma) or not 0<=gamma<=1:
            raise ValueError("gamma must be between 0 and 1")
        self.scale=float(scale)
        self.gamma=float(gamma)
        self._previous_phi=None

    def _potential(self,info):
        # Getting closer increases the potential, before and after 45 seconds
        return -float(info["distance_to_safehouse"])/WORLD_DIAGONAL

    def _reward_info(self,info,base_reward,shaping_reward):
        # Copy dictionaries so the original environment's info is not changed
        result=dict(info)
        result["base_reward"]=float(base_reward)
        result["shaping_reward"]=float(shaping_reward)
        result["training_reward"]=float(base_reward+shaping_reward)
        result["reward_terms"]=dict(info.get("reward_terms",{}))
        result["reward_terms"]["goal_potential"]=float(shaping_reward)
        return result

    def reset(self,*,seed=None,options=None):
        observation,info=self.env.reset(seed=seed,options=options)
        self._previous_phi=self._potential(info)
        return observation,self._reward_info(info,0.0,0.0)

    def step(self,action):
        if self._previous_phi is None:
            raise gym.error.ResetNeeded("Call reset() before stepping")
        observation,base_reward,terminated,truncated,info=self.env.step(action)
        # A real terminal state has zero potential. A time limit still bootstraps.
        next_phi=0.0 if terminated else self._potential(info)
        shaping_reward=self.scale*(self.gamma*next_phi-self._previous_phi)
        self._previous_phi=next_phi
        info=self._reward_info(info,base_reward,shaping_reward)
        return observation,info["training_reward"],terminated,truncated,info
