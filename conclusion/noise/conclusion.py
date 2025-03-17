import pandas as pd
import matplotlib.pyplot as plt
import os
from cycler import cycler
import re

base_dir = './conclusion/noise'
files = [f for f in os.listdir(base_dir) if re.search(r'\d+$', f) and (not f.startswith('0'))]
# 新增排序逻辑：提取文件名中的数值部分进行排序
files = sorted(files, key=lambda x: float(x.split(',')[0])) 
colors = plt.cm.tab10.colors  
linestyles = ['-', '--', '-.', ':'] * 3  
style_cycler = cycler(color=colors) + cycler(linestyle=linestyles[:len(colors)])


fig, (ax1,ax2) = plt.subplots(2,1,figsize=(12,10))
ax1.set_prop_cycle(style_cycler)
ax2.set_prop_cycle(style_cycler)

all_rewards = []
all_success = []
for file in files:
    log_path = os.path.join(base_dir, file, 'logs', 'mw-door-unlock', 'state', 'default', '10', 'eval.log')
    df = pd.read_csv(log_path)
    ax1.plot(df['env_step'],df['episode_reward'], label = file,linewidth=2,alpha=0.8)
    ax2.plot(df['env_step'],df['episode_success'], label = file,linewidth=2,alpha=0.8)

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
output_path = os.path.join(base_dir, 'conclusion-1.png')
plt.tight_layout()
plt.savefig(output_path)
plt.close()  # 关闭图形以释放内存