import copy
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter
from tianshou.data.buffer.base import ReplayBuffer
from tianshou.data.batch import Batch


class BasePolicy:

    def __init__(self,
                 gamma: float=0.98,
                 buffer_size: int=100000,
                 log_name: str="",
                 ):
        
        self.gamma = gamma
        self.buffer = ReplayBuffer(buffer_size)
        
        if log_name:
            self.writer=SummaryWriter(f"./logs/{log_name}")
        else:
            raise ValueError("log_name must be specified")
    
    def choose_action(self, state):
        """_summary_

        Args:
            state: the state of the environment

        Returns:
            the action chosen by the policy
        """
        raise NotImplementedError
    
    def update(self):
        raise NotImplementedError
    
    def store(self, state, action, reward, next_state, terminated, truncated, info):
        batch = Batch({
            "obs": state,
            "act": action,
            "rew": reward,
            "terminated": terminated,
            "obs_next": next_state,
            "truncated": truncated,
            "info": info,
        })
        self.buffer.add(batch)
        
    def clear_buffer(self):
        self.buffer.reset()        