from QTransformer import QTransformer
import torch
import torch.nn as nn
from typing import Dict, List
import numpy as np
from copy import deepcopy
from utils import ema


class QTransformerServer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.misc.device)
        
        # 主Q网络
        self.q_transformer = QTransformer(cfg.qtransformer).to(self.device)
        # 目标网络
        self.target_network = deepcopy(self.q_transformer).requires_grad_(False)
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            self.q_transformer.parameters(),
            lr=cfg.qtransformer.lr,
            eps=cfg.qtransformer.eps,
            weight_decay=cfg.qtransformer.decay
        )
        
        self.total_steps = 0
        
        # 添加学习率调度
        if cfg.env.domain == "metaworld":
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lambda steps: (1 - steps/cfg.env.train_steps) if cfg.distributed.num_agents == 1  # n=1时线性衰减
                else 1 / (1 + np.log(cfg.distributed.num_agents) * (steps / cfg.env.train_steps))  # n>1时对数衰减
            )
        # # 修改后的线性衰减策略
        # if cfg.env.domain == "metaworld":
        #     self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        #         self.optimizer,
        #         lambda steps: max(1 - (steps * cfg.distributed.num_agents) / cfg.env.train_steps, 0)
        #     )
        
    def update_networks(self, batch_experiences: List[Dict]):
        """更新网络,处理多个agent的经验"""
        metrics = dict()
        
        # 合并所有agent的经验
        combined_batch = self._combine_experiences(batch_experiences)
        
        # 确保数据在正确的设备上
        combined_batch = {
            k: v.to(self.device) if torch.is_tensor(v) else v 
            for k, v in combined_batch.items()
        }
        
        # 训练模式
        self.q_transformer.train()
        self.target_network.train()
        
        # 清空梯度
        self.optimizer.zero_grad(set_to_none=True)
        
        # 计算损失
        loss, td_loss, conservative_loss = self.q_transformer.compute_loss(
            combined_batch,
            self.target_network
        )
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        if self.cfg.qtransformer.grad_clip is not None:
            grad_norm = nn.utils.clip_grad_norm_(
                self.q_transformer.parameters(),
                self.cfg.qtransformer.grad_clip
            )
        
        # 优化器步骤
        self.optimizer.step()
        
        # 添加学习率调度步骤
        if self.cfg.env.domain == "metaworld":
            self.lr_scheduler.step()
        
        # 更新目标网络
        if self.total_steps % self.cfg.qtransformer.update_freq == 0:
            ema(self.q_transformer, self.target_network, self.cfg.qtransformer.tau)
        
        self.total_steps += 1

        # # 新增学习率调整逻辑
        # if self.total_steps == 180000:
        #     num_agents = self.cfg.distributed.num_agents
        #     new_lr = self.optimizer.param_groups[0]['lr'] / num_agents
        #     self.optimizer.param_groups[0]['lr'] = new_lr
        #     print(f"Adjusted learning rate to {new_lr} at step {self.total_steps}")
        
        # 评估模式
        self.q_transformer.eval()
        self.target_network.eval()
        
        # 记录指标
        metrics = {
            'loss': loss.item(),
            'td_loss': td_loss.item(),
            'conservative_loss': conservative_loss.item(),
            'grad': grad_norm.item(),
            'lr': self.optimizer.param_groups[0]['lr']
        }
        
        return {
            'network_state': self.q_transformer.state_dict(),
            'metrics': metrics
        }
        
    def _combine_experiences(self, batch_experiences: List[Dict]) -> Dict:
        """合并多个agent的经验"""
        combined = {}
        # 确保有数据
        if not batch_experiences:
            return combined
            
        # 获取所有键
        keys = batch_experiences[0].keys()
        
        # 对每个键合并数据
        for key in keys:
            # 收集所有agent的该键数据
            tensors = [b[key] for b in batch_experiences]
            # 在第一维度(batch维度)上拼接
            combined[key] = torch.cat(tensors, dim=0)
            
        return combined
        
    def get_state_dict(self):
        """获取当前网络状态"""
        return self.q_transformer.state_dict() 

