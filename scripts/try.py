import matplotlib.pyplot as plt
import numpy as np
from math import log

T = 1e6  # 总训练步数
n_values = [1, 2, 4, 8]
t = np.linspace(0, T, 1000)

plt.figure(figsize=(8,5))
for n in n_values:
    decay_rate = log(n) if n > 1 else 0
    lr = np.maximum(1 - (decay_rate * t)/T, 0)
    plt.plot(t/T, lr, label=f'n={n}', lw=2)

plt.title("基于log(n)系数的线性衰减")
plt.xlabel('训练进度 (t/T)')
plt.ylabel('学习率比例 (lr/lr0)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()