# 🧠 Distributed Q-Transformer (DQT)
**Implementation of “Scalability and Noise Resilience in Q-Transformer” (Wang, 2024)**
**分布式 Q-Transformer：可扩展性与鲁棒性研究实现**
📄 [论文链接](https://openreview.net/pdf?id=WQupWGepAO)
---

## 📘 Overview | 项目简介

This project is built upon [lucidrains/q-transformer](https://github.com/lucidrains/q-transformer),

extending its core architecture to a distributed multi-agent setting with enhanced noise robustness and scalability analysis.

---

## 🚀 Quick Start | 快速开始

### 1️⃣ Setup Environment / 环境配置
```bash
git clone https://github.com/<username>/Distributed-QT.git
cd Distributed-QT
conda env create -f environment.yml
conda activate dqt
```

### 2️⃣ Run Distributed Training / 运行分布式训练
```bash
python src/main.py --num_agents 4 --train_steps 300000 --task mw-door-unlock
```

### 3️⃣ Run Noise Robustness Test / 噪声鲁棒性测试
```bash
python scripts/run_noise_experiments.py --noise_mean 0 --noise_variance 1.0
```

All logs and visual outputs will be generated under `experiment_logs/` and `outputs/`.

---

## 📂 Directory Structure | 项目结构

```
DISTRIBUTED-QT/
├── src/
│   ├── distributed/     # 分布式核心模块（agent / server / trainer）
│   ├── QTransformer.py  # 主模型定义
│   ├── main.py          # 训练入口
│   └── config.yaml      # 配置文件
├── scripts/             # 实验脚本与可视化
├── conclusion/          # 实验结论与图表
├── experiment_logs/     # 日志与指标
├── outputs/             # 输出结果
└── environment.yml
```

---

## 🧪 Experiments | 实验简述

We conducted controlled experiments on **MetaWorld Door-Unlock**, testing scalability across multiple agents and evaluating robustness under noisy rewards.
本研究在 **MetaWorld Door-Unlock 任务** 上进行系统测试，涉及不同智能体数量和噪声环境设置。

- **Distributed Training:**
  Compared configurations with 1, 4, 7, and 11 agents under identical settings.
  Multi-agent setups (esp. 7 agents) showed **faster convergence** and **more stable reward curves** than single-agent baselines.

- **Noise Robustness:**
  Added Gaussian noise (mean 0–10, variance 0.1–100) to rewards.
  Median-based filtering effectively reduced performance degradation under moderate noise (variance ≤ 1.0).

- **Learning Rate Schedules:**
  Evaluated linear, log curve, and adaptive decay strategies.
  Log-based decay yielded **smooth convergence** across varying agent counts.

Overall, DQT achieved **higher stability and sample efficiency**, confirming the effectiveness of distributed and robust training strategies.
总体结果表明，分布式 Q-Transformer 在**收敛速度**、**稳定性**及**鲁棒性**方面均优于单智能体。

---


✅ **Summary / 总结：**
Distributed Q-Transformer demonstrates **efficient scalable training** and **robust performance under noise**, offering a practical framework for real-world reinforcement learning.
分布式 Q-Transformer 在**可扩展训练与噪声鲁棒性**方面表现优越，为强化学习的工程化应用提供了可行范式。
