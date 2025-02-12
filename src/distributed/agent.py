from copy import deepcopy
import torch
import numpy as np
from QTransformer import QTransformer
from utils import LinearSchedule


class DistributedAgent:
    def __init__(self, cfg, agent_id):
        self.cfg = cfg
        self.agent_id = agent_id
        self.device = torch.device(cfg.misc.device)
        
        # 本地Q网络
        self.q_transformer = QTransformer(cfg.qtransformer).to(self.device)
        
        # 探索策略
        self.exploration = LinearSchedule(
            self.cfg.env.train_steps * 0.5,
            0.01,
            initial_p=1.0
        )
        
    def update_network(self, network_state):
        """从服务器更新网络参数"""
        self.q_transformer.load_state_dict(network_state)
        
    @torch.no_grad()
    def get_action(self, obs, step=None, eval_mode=False):
        """选择动作"""
        if step is None:
            step = 0
            
        # 随机动作阶段
        if not eval_mode and step < self.cfg.misc.seed_steps:
            return self.q_transformer.get_random_action(1).squeeze(0).cpu()
            
        # epsilon-greedy探索
        if not eval_mode:
            epsilon = self.exploration.value(step)
            if np.random.random() < epsilon:
                action = self.q_transformer.get_random_action(1).squeeze(0)
                return action.cpu()
                
        # 正常动作选择
        obs = torch.FloatTensor(obs).to(self.device)
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)
        with torch.no_grad():
            action = self.q_transformer.get_optimal_actions(obs)
            return action.cpu() 