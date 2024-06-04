import os
import torch
import collections
import numpy as np
from torch import nn, optim
from itertools import chain
import torch.nn.functional as F
from torch.distributions import Categorical
from .policy import BasePolicy
from utils.utils import discount_cum, minimize_with_clipping

class PPO(BasePolicy):
    
    def __init__(self, 
                 pi_net: nn.Module,
                 v_net: nn.Module,
                 lam: float=0.95,
                 clip_ratio: float=0.2,
                 lr: float=3e-4,
                 max_global_gradient_norm: float = 0.5, 
                 gamma: float = 0.99, 
                 critic_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 buffer_size: int = 100000, 
                 batch_size: int = 64,
                 batch_num: int = 0,
                 adam_eps: float = 1e-5,
                 normalize_adv: bool = True,
                 log_name: str = ""):
        
        super().__init__(gamma, buffer_size, log_name)
        self.pi_net = pi_net
        self.v_net = v_net
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.critic_coef = critic_coef
        self.entropy_coef = entropy_coef
        self.batch_size = batch_size
        self.max_global_gradient_norm = max_global_gradient_norm
        self.normalize_adv = normalize_adv
        self.batch_num = batch_num
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr, eps=adam_eps)
        self.dataset = collections.defaultdict(list)
        
        assert next(self.pi_net.parameters()).device == next(self.v_net.parameters()).device
        self.device = next(self.pi_net.parameters()).device
    
    def choose_action(self, state):
        with torch.no_grad():
            obs = torch.FloatTensor(state).to(self.device)
            act_dist = self.pi_net(obs)
            act = act_dist.sample().cpu().numpy()
        return act
    
    def update(self, update_iters=10):
        self.buffer_to_dataset()
        count = 0
        tot_pi_loss = 0
        tot_v_loss = 0
        tot_entropy = 0
        tot_ratio = 0
        tot_fraction = 0
        
        if self.batch_num:
            batch_size = len(self.dataset["obs"]) // self.batch_num
        else:
            batch_size = min(self.batch_size, len(self.dataset["obs"]))
        for i in range(update_iters):
            for _ in range(len(self.dataset["obs"])//self.batch_size + 1):
                batch_idx = np.random.choice(len(self.dataset["obs"]), size=batch_size)
                batch = {key: np.array(self.dataset[key])[batch_idx] for key in self.dataset}
                    
                pi_loss, ratio, fraction = self.pi_loss(batch)
                v_loss = self.v_loss(batch)
                entropy = self.entropy(batch)
                loss = pi_loss + self.critic_coef * v_loss - self.entropy_coef * entropy
                minimize_with_clipping(self.parameters(), self.optimizer, loss, self.max_global_gradient_norm)
                
                tot_pi_loss += pi_loss.item()
                tot_v_loss += v_loss.item()
                tot_entropy += entropy.item()
                tot_ratio += ratio.item()
                tot_fraction += fraction
                count += 1
        self.clear_buffer()
        return {"train/pi_loss": tot_pi_loss/count, 
                "train/v_loss": tot_v_loss/count, 
                "train/entropy": tot_entropy/count,
                "train/ratio": tot_ratio/count,
                "train/fraction": tot_fraction/count,
                }
    
    def pi_loss(self, batch):
        
        device = self.device
        obs = torch.FloatTensor(np.array(batch["obs"])).to(device)
        act = torch.FloatTensor(np.array(batch["act"])).to(device)
        adv = torch.FloatTensor(np.array(batch["adv"])).to(device)
        log_p_old = torch.FloatTensor(np.array(batch["log_p"])).to(device)
        
        if self.normalize_adv:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        
        log_p = self.pi_net(obs).log_prob(act)
        
        if log_p.dim() > 1:
            adv = adv.unsqueeze(-1)
        
        ratio = torch.exp(log_p - log_p_old)
        loss = ratio * adv
        clip_loss = torch.clip(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
        fraction = (clip_loss == loss).sum().item() / len(loss.flatten())
        loss = torch.min(loss, clip_loss)
        loss = -torch.mean(loss)
        
        return loss, ratio.mean(), fraction
    
    def v_loss(self, batch):
        obs = torch.FloatTensor(np.array(batch["obs"])).to(self.device)
        ret = torch.FloatTensor(np.array(batch["return"])).to(self.device).unsqueeze(-1)
        
        val = self.v_net(obs)
        loss = F.mse_loss(val, ret)
        return loss
     
    def entropy(self, batch):
        obs = torch.FloatTensor(np.array(batch["obs"])).to(self.device)
        act_dist = self.pi_net(obs)
        entropy = act_dist.entropy()
        return entropy.mean() 
    
    def buffer_to_dataset(self):
        batch = self.buffer.sample(0)[0]
        obs = batch["obs"]
        act = batch["act"]
        rew = batch["rew"]
        done = batch["done"]
        terminated = batch["terminated"]
        obs_next = batch["obs_next"]
        ret = discount_cum(rew, done, self.gamma)
        
        # GAE-Lambda advantage calculation
        adv = np.zeros_like(rew)
        with torch.no_grad():
            obs_ = torch.FloatTensor(obs).to(self.device)
            obs_next_ = torch.FloatTensor(obs_next).to(self.device)
            vals = self.v_net(obs_).cpu().numpy().reshape(-1,)
            vals_next = self.v_net(obs_next_).cpu().numpy().reshape(-1,)
            adv = rew + self.gamma * vals_next * (1 - terminated) - vals
        adv = discount_cum(adv, done, self.gamma * self.lam)
        
        ret = adv + vals
        
        # computing log pi
        with torch.no_grad():
            obs_ = torch.FloatTensor(obs).to(self.device)
            act_ = torch.FloatTensor(act).to(self.device)
            log_p = self.pi_net(obs_).log_prob(act_).cpu().numpy()
        
        self.dataset["obs"].extend(obs)
        self.dataset["act"].extend(act)
        self.dataset["rew"].extend(rew)
        self.dataset["return"].extend(ret)
        self.dataset["done"].extend(done)
        self.dataset["obs_next"].extend(obs_next)
        self.dataset["adv"].extend(adv)
        self.dataset["log_p"].extend(log_p)
    
    def clear_buffer(self):
        super().clear_buffer()
        self.dataset = collections.defaultdict(list)
    
    def parameters(self):
        return chain(self.pi_net.parameters(), self.v_net.parameters())
    
    def save(self, path, post_fix=""):
        torch.save(
            {
                "pi_net": self.pi_net.state_dict(),
                "v_net": self.v_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            os.path.join(path, "ppo_{}.pth.tar".format(post_fix))
        )
        pass
    
    def load(self, path, post_fix=""):
        model_ckpt = torch.load(os.path.join(path, "ppo_{}.pth.tar".format(post_fix)))
        self.pi_net.load_state_dict(model_ckpt["pi_net"])
        self.v_net.load_state_dict(model_ckpt["v_net"])
        self.optimizer.load_state_dict(model_ckpt["optimizer"])
