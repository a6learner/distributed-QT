import matplotlib.pyplot as plt
from math import log
import numpy as np
def lr_schedule(t, T, n):
    """复合衰减策略"""
    if n == 1:
        return 1 - t/T  # 线性衰减
    else:
        return 1 / (1 + log(n) * (t/T))  # 对数衰减

# 参数设置
T_options = [1e5, 1e6]  # 总步数选项
n_options = [1, 4, 8]   # agent数量选项

plt.figure(figsize=(12, 5))

# 子图1：不同agent数量的衰减曲线
plt.subplot(1, 2, 1)
T = 1e6
for n in n_options:
    t = np.linspace(0, T, 1000)
    lr = [lr_schedule(step, T, n) for step in t]
    plt.plot(t/T, lr, label=f'n={n}', lw=2)
    
plt.title(f"不同Agent数量的衰减曲线 (T={T})")
plt.xlabel('训练进度 (t/T)')
plt.ylabel('学习率比例 (lr/lr0)')
plt.grid(True, alpha=0.3)
plt.legend()

# 子图2：不同总步数的影响
plt.subplot(1, 2, 2)
n = 4
for T in T_options:
    t = np.linspace(0, T, 1000)
    lr = [lr_schedule(step, T, n) for step in t]
    plt.plot(t/T, lr, label=f'T={int(T)}', lw=2)

plt.title(f"不同总步数的影响 (n={n})")
plt.xlabel('训练进度 (t/T)')
plt.ylabel('学习率比例 (lr/lr0)')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show() 