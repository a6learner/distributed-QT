import warnings
warnings.filterwarnings('ignore')
import os
import sys
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
import random
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from pathlib import Path
from logger import Logger, make_dir
from utils import ReplayBuffer, Episode
from envs import dmc_env, meta_env
from distributed.server import QTransformerServer
from distributed.agent import DistributedAgent
import math

__LOGS__ = 'logs'
__MEDIA__ = 'media'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class DistributedTrainer:
    def __init__(self, cfg):
        assert torch.cuda.is_available()
        self.cfg = cfg
        self.cfg.qtransformer.lr /= self.cfg.distributed.num_agents
        set_seed(cfg.misc.seed)
        self.device = torch.device(cfg.misc.device)
        self.work_dir = Path().cwd() / __LOGS__ /cfg.env.task / cfg.env.modality / cfg.wandb.exp_name / str(cfg.misc.seed)
        self.reconstructions_dir = make_dir(self.work_dir / __MEDIA__ / 'reconstructions')
        make_dir(self.reconstructions_dir)
        

        
        # 创建评估环境(create the evaluation enviroment)
        if cfg.env.domain=='dmc_suite' or cfg.env.domain=='dmc_manip':
            self.env = dmc_env.make_env(cfg.env)
        elif cfg.env.domain=='metaworld':
            self.env = meta_env.make_env(cfg.env)
        else:
            print("env.domain takes either metaworld or dmc_suite")
            
        # 保存原始train_stepssave the origin train_steps)
        self.original_train_steps = cfg.env.train_steps

        # 创建服务器和多个agent(create server and several agents )
        # change the lr
        # if self.cfg.distributed.num_agents > 1:
        #     self.cfg.qtransformer.lr = math.log(self.cfg.distributed.num_agents) * self.cfg.qtransformer.lr
        self.server = QTransformerServer(self.cfg)
        self.agents = []
        for i in range(cfg.distributed.num_agents):
            # 每次创建环境前恢复原始train_steps(restore the train_steps before create agents)
            cfg.env.train_steps = self.original_train_steps
            agent = DistributedAgent(cfg, i)
            buffer = ReplayBuffer(cfg.env)
            env = dmc_env.make_env(cfg.env) if cfg.env.domain=='dmc_suite' else meta_env.make_env(cfg.env)
            self.agents.append({
                'agent': agent,
                'buffer': buffer,
                'env': env,
                'steps_done': 0  # 记录每个agent的训练步数
            })

        # 最后再次恢复原始train_steps
        cfg.env.train_steps = self.original_train_steps
        print(f'{sum(p.numel() for p in self.server.q_transformer.parameters())} parameters in server.Q_transformer')
    
    def run(self):
        L = Logger(self.work_dir, self.cfg)
        print(self.cfg.qtransformer.lr)
        episode_idx, start_time = 0, time.time()
        print("Start Training")
        print("===============")
        
        for step in range(0, self.cfg.env.train_steps+self.cfg.env.episode_length, self.cfg.env.episode_length):
            # 收集所有agent的经验
            batch_experiences = []
            metrics = {}
            
            for agent_info in self.agents:
                agent = agent_info['agent']
                env = agent_info['env']
                buffer = agent_info['buffer']

                # 收集轨迹
                obs = env.reset()
                episode = Episode(self.cfg.env, obs)
                t = 0
                
                while not episode.done:
                    action = agent.get_action(obs, step=step+t)
                    # convert the discretize action into continuous before applying it to env
                    env_action = (action.numpy()/self.cfg.qtransformer.num_bins*2-1) 
                    obs, reward, done, info = env.step(env_action)
                    episode += (obs, action, reward, done, info['success'])
                    t += 1

                # 添加episode到buffer
                buffer += episode
                episode_idx += 1
                env_step = int(step*self.cfg.env.action_repeat)

                # 采样经验用于更新
                if step >= self.cfg.misc.seed_steps:
                    batch = buffer.sample_batch(
                        self.cfg.qtransformer.batch_size,
                        self.cfg.qtransformer.n_step_td + 1
                    )
                    batch_experiences.append(batch)

            # 服务器更新网络
            if step >= self.cfg.misc.seed_steps and batch_experiences:
                num_updates = self.cfg.env.episode_length 
                for i in tqdm(range(num_updates), desc=f"Training {str(self.server)}", file=sys.stdout):
                    update = self.server.update_networks(batch_experiences)                   
                    # 更新所有agent的网络
                    for agent_info in self.agents:
                        agent_info['agent'].update_network(update['network_state'])                    
                    metrics.update(update['metrics'])
            
            common_metrics = {
                'episode': episode_idx,
                'step': step,
                'env_step': env_step,
                'total_time': time.time() - start_time,
                'episode_reward': episode.cumulative_reward,
                'episode_success': episode.success,
            }
            metrics.update(common_metrics)
            L.log(metrics, category='train')

            # 定期评估
            if env_step % self.cfg.misc.eval_freq == 0:
                common_metrics['episode_reward'], common_metrics['episode_success'] = self.evaluate(self.cfg.misc.eval_episodes, step, env_step, L.video)
                L.log(common_metrics, category='eval')
                
        L.finish(self.server)
        print('Training completed successfully')

    @torch.no_grad()
    def evaluate(self, num_episodes, step, env_step, video):
        """评估阶段"""
        episode_rewards, episode_success = [], []
        for i in range(num_episodes):
            obs, done, ep_reward, t = self.env.reset(), False, 0, 0
            if video: video.init(self.env, enabled=(i==0))
            while not done:
                action = self.agents[0]['agent'].get_action(obs, step=step+t, eval_mode=True)
                # convert the discretize action into continuous before applying it to env
                env_action = action/self.cfg.qtransformer.num_bins*2-1 
                obs, reward, done, info = self.env.step(env_action.cpu().numpy())
                ep_reward += reward
                if video: video.record(self.env)
                t += 1
            episode_rewards.append(ep_reward)
            episode_success.append(info['success'])
            if video: video.save(env_step)
        return np.nanmean(episode_rewards), np.nanmean(episode_success) 