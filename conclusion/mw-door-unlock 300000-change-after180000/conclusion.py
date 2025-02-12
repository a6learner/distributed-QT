import pandas as pd
import matplotlib.pyplot as plt
import os
from cycler import cycler

# 获取当前目录下的所有日志文件
base_dir = './conclusion/mw-door-unlock 300000-change-after180000'
target_agents = [1, 4, 7, 11]
P_files = [f for f in os.listdir(base_dir) if not f.endswith(".py")]
all_agent = ['agent1', 'agent4', 'agent7', 'agent11'] 
# 设置颜色和线型的循环样式
colors = plt.cm.tab10.colors  # 选取一个颜色调色板
# linestyles = ['-', '--', '-.', ':']
style_cycler = cycler(color=colors) 
# * cycler(linestyle=linestyles)

# 设置子图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# 应用样式循环
ax1.set_prop_cycle(style_cycler)
ax2.set_prop_cycle(style_cycler)



for file in all_agent:
    all_rewards = []
    all_success = []    
    for files in P_files:
        log_path = os.path.join(base_dir, files, file, 'logs', 'mw-door-unlock', 'state', 'default', '10', 'eval.log')
        df = pd.read_csv(log_path)
        all_rewards.append(df['episode_reward'])
        all_success.append(df['episode_success'])
        # 合并数据，计算平均
    avg_reward = pd.concat(all_rewards, axis=1).mean(axis=1)
    avg_success = pd.concat(all_success, axis=1).mean(axis=1)

    ax1.plot(df['env_step'], avg_reward, label=file, linewidth=2, alpha=0.8)
    ax2.plot(df['env_step'], avg_success, label=file, linewidth=2, alpha=0.8)

# 图表美化
# 图1: Episode Reward vs Env Step
ax1.set_title('Episode Reward vs Environment Step', fontsize=16)
ax1.set_xlabel('Environment Step', fontsize=14)
ax1.set_ylabel('Episode Reward', fontsize=14)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(title='Files', fontsize=10, title_fontsize=12, loc='best')

# 图2: Episode Success vs Env Step
ax2.set_title('Episode Success vs Environment Step', fontsize=16)
ax2.set_xlabel('Environment Step', fontsize=14)
ax2.set_ylabel('Episode Success', fontsize=14)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(title='Files', fontsize=10, title_fontsize=12, loc='best')

# 调整布局并保存图形
output_path = os.path.join(base_dir, 'conclusion.png')
plt.tight_layout()
plt.savefig(output_path)
plt.close()  # 关闭图形以释放内存
