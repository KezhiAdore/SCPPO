import sys
import copy
from torch.nn.modules import Module
sys.path.append('.')
import torch
from torch import nn
import numpy as np
import collections
from torch.nn import functional as F
from .ppo import PPO
from utils.utils import v_trace, minimize_with_clipping, discount_cum, weight_discount_cum

class SCPPO(PPO):
    def __init__(self, 
                 pi_net: nn.Module, 
                 v_net: nn.Module, 
                 lam: float = 0.95, 
                 clip_ratio: float = 0.2, 
                 lr: float = 3e-4, 
                 max_global_gradient_norm: float = 0.5, 
                 gamma: float = 0.99, 
                 critic_coef: float = 0.5, 
                 entropy_coef: float = 0.00, 
                 buffer_size: int = 10000, 
                 batch_size: int = 64, 
                 batch_num: int = 0,
                 adam_eps: float = 1e-5,
                 normalize_adv: bool = True, 
                 log_name: str = "", 
                 history_len: int = 10,
                 a_trace: bool = False,
                 target_kl: float = 0.01,
                 beta: float = 1.0,
                 ):
        super().__init__(pi_net, v_net, lam, clip_ratio, lr, 
                         max_global_gradient_norm, gamma, critic_coef, 
                         entropy_coef, buffer_size, batch_size, batch_num, 
                         adam_eps, normalize_adv, log_name)
        self.policy_history = []
        self.dataset_buffer = []
        self.sample_buffer = collections.defaultdict(list)
        self.history_count = 0
        
        self.history_len = history_len
        self.a_trace = a_trace
        self.target_kl = target_kl
        self.beta = beta
        
    def buffer_to_dataset(self):
        batch = self.buffer.sample(0)[0]
        obs = batch["obs"]
        act = batch["act"]
        rew = batch["rew"]
        done = batch["done"]
        obs_next = batch["obs_next"]
        terminated = batch["terminated"]
        truncated = batch["truncated"]
        # ret = discount_cum(rew, done, self.gamma)
        
        # computing log pi & original advantage
        with torch.no_grad():
            obs_ = torch.FloatTensor(obs).to(self.device)
            act_ = torch.FloatTensor(act).to(self.device)
            obs_next_ = torch.FloatTensor(obs_next).to(self.device)
            log_p = self.pi_net(obs_).log_prob(act_).cpu().numpy()
            vals = self.v_net(obs_).cpu().numpy().reshape(-1,)
            vals_next = self.v_net(obs_next_).cpu().numpy().reshape(-1,)
            adv = rew + self.gamma * vals_next * (1 - terminated) - vals
        
        self.dataset["obs"].extend(obs)
        self.dataset["act"].extend(act)
        self.dataset["rew"].extend(rew)
        self.dataset["done"].extend(done)
        self.dataset["terminated"].extend(terminated)
        self.dataset["truncated"].extend(truncated)
        self.dataset["obs_next"].extend(obs_next)
        self.dataset["log_p"].extend(log_p)
        self.dataset["adv_origin"].extend(adv)
        
        if len(self.dataset_buffer) < self.history_len:
            self.dataset_buffer.append(self.dataset)
            self.policy_history.append(copy.deepcopy(self.pi_net))
        else:
            self.dataset_buffer.pop(0)
            self.policy_history.pop(0)
            self.dataset_buffer.append(self.dataset)
            self.policy_history.append(copy.deepcopy(self.pi_net))
        self.history_count += 1
        self.clear_buffer()
        self.update_dataset_buffer()
    
    def update_dataset_buffer(self):
        self.sample_buffer = collections.defaultdict(list)
        # compute v_trace
        for i, dataset in enumerate(self.dataset_buffer):
            for key in dataset:
                self.sample_buffer[key].extend(dataset[key])
        obs = torch.FloatTensor(np.array(self.sample_buffer["obs"])).to(self.device)
        act = torch.FloatTensor(np.array(self.sample_buffer["act"])).to(self.device)
        with torch.no_grad():
            log_p = self.pi_net(obs).log_prob(act)
            self.sample_buffer["log_p_last"] = log_p.cpu().numpy()
        
        self.sample_buffer = self.update_dataset(self.sample_buffer)
        
    def update_dataset(self, dataset):
        device = self.device
        obs = np.array(dataset["obs"])
        obs_next = np.array(dataset["obs_next"])
        act = np.array(dataset["act"])
        log_p_i = np.array(dataset["log_p"])
        rew = np.array(dataset["rew"])
        done = np.array(dataset["done"])
        terminated = np.array(dataset["terminated"])
        truncated = np.array(dataset["truncated"])
        
        obs_ = torch.FloatTensor(obs).to(device)
        obs_next_ = torch.FloatTensor(obs_next).to(device)
        act_ = torch.FloatTensor(act).to(device)
        log_p_i_ = torch.FloatTensor(log_p_i).to(device)
        rew_ = torch.FloatTensor(rew).unsqueeze(-1).to(device)
        terminated_ = torch.FloatTensor(terminated).unsqueeze(-1).to(device)
        done_ = torch.FloatTensor(done).unsqueeze(-1).to(device)
        
        # compute v_trace
        with torch.no_grad():
            log_p_ = self.pi_net(obs_).log_prob(act_)
            if log_p_.dim() == 1:
                log_rho = (log_p_ - log_p_i_).unsqueeze(-1)
            else:
                log_rho = log_p_.sum(1, True) - log_p_i_.sum(1, True)
            val_ = self.v_net(obs_).to(device)
            val_next_ = self.v_net(obs_next_).to(device)
        vs_list = []
        done_idx = np.where(done)[0]
        
        if (len(done) - 1) not in done_idx:
            done_idx = np.append(done_idx, len(done) - 1)
        done_idx = np.insert(done_idx, 0, -1)
        
        for j in range(len(done_idx) - 1):
            start = done_idx[j] + 1
            end = done_idx[j+1] + 1
            if terminated[end - 1]:
                bootstrap_value = torch.zeros_like(val_[0])
            else:
                bootstrap_value = val_next_[end-1]
            vs = v_trace(log_rho[start:end], self.gamma, rew_[start:end], 
                         val_[start:end], bootstrap_value)
            vs_list.append(vs)
        vs = torch.cat(vs_list)
        dataset["vs"] = vs.cpu().numpy()
        
        # advantage calculation
        adv_ = rew_ + self.gamma * val_next_ * (1 - terminated_) - val_
        
        # advantage correction
        if self.a_trace:
            weight = torch.exp(log_rho)
            weight = torch.clip(weight, 0, 1)
            adv_ = weight_discount_cum(adv_, done, weight, self.gamma * self.lam)

        dataset["adv"] = adv_.squeeze(-1).cpu().numpy()

        return dataset
        
    def update(self, update_iters=10):
        self.buffer_to_dataset()
        count = 0
        tot_pi_loss = 0
        tot_v_loss = 0
        tot_entropy = 0
        tot_ratio = 0
        tot_fraction = 0
        tot_approx_kl = 0
        
        if self.batch_num:
            batch_size = len(self.sample_buffer["obs"]) // self.batch_num
        else:
            batch_size = self.batch_size
            
        for i in range(update_iters):
            for _ in range(len(self.sample_buffer["obs"])//batch_size + 1):
                batch_idx = np.random.choice(len(self.sample_buffer["obs"]), size=batch_size)
                batch = {key: np.array(self.sample_buffer[key])[batch_idx] for key in self.sample_buffer}
                pi_loss, ratio, fraction, approx_kl = self.pi_loss(batch)
                v_loss = self.v_loss(batch)
                entropy = self.entropy(batch)
                loss = pi_loss + self.critic_coef * v_loss - self.entropy_coef * entropy
                minimize_with_clipping(self.parameters(), self.optimizer, loss, self.max_global_gradient_norm)
                
                tot_pi_loss += pi_loss.item()
                tot_v_loss += v_loss.item()
                tot_entropy += entropy.item()
                tot_ratio += ratio.item()
                tot_fraction += fraction
                tot_approx_kl += approx_kl.item()
                count += 1
        return {"train/pi_loss": tot_pi_loss/count, 
                "train/v_loss": tot_v_loss/count, 
                "train/entropy": tot_entropy/count,
                "train/ratio": tot_ratio/count,
                "train/fraction": tot_fraction/count,
                "train/approx_kl": tot_approx_kl/count,
                }
    
    def pi_loss(self, batch):
        device = self.device
        obs = torch.FloatTensor(np.array(batch["obs"])).to(device)
        act = torch.FloatTensor(np.array(batch["act"])).to(device)
        adv = torch.FloatTensor(np.array(batch["adv"])).to(device)
        log_p_i = torch.FloatTensor(batch["log_p"]).to(device)
        log_p_last = torch.FloatTensor(batch["log_p_last"]).to(device)
        
        if self.normalize_adv:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        
        log_p = self.pi_net(obs).log_prob(act)
        
        if log_p.dim() > 1:
            adv = adv.unsqueeze(-1)
        
        last_ratio = log_p - log_p_last
        last_ratio = torch.clip(last_ratio, -10, 10)
        approx_kl = torch.mean(torch.exp(last_ratio) - 1 - last_ratio)
        
        ratio = torch.exp(log_p - log_p_i)
        
        loss = ratio * adv
        clip_loss = torch.clip(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
        
        fraction = (clip_loss == loss).sum().item() / len(loss.flatten())
        
        if self.target_kl is not None:
            if approx_kl > 1.5 * self.target_kl:
                self.beta = 2 * self.beta
            elif approx_kl < self.target_kl / 1.5:
                self.beta = self.beta / 2
        
        self.beta = np.clip(self.beta, 0.1, 10)
        
        loss = torch.min(loss, clip_loss) 
        loss = -torch.mean(loss) + self.beta * approx_kl
        
        return loss, ratio.mean(), fraction, approx_kl
    
    def v_loss(self, batch):
        obs = torch.FloatTensor(np.array(batch["obs"])).to(self.device)
        vs = torch.FloatTensor(np.array(batch["vs"])).to(self.device)
        
        val = self.v_net(obs)
        loss = F.mse_loss(val, vs)
        return loss
