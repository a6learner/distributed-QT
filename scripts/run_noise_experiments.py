import subprocess
from pathlib import Path
import itertools
import time
import yaml
import os
import torch

def run_noise_experiment(params, log_dir):
    """运行单次噪声实验"""
    cmd = [
        'python',
        'src/main.py',
        f"noise.mean={params['mean']}",
        f"noise.variance={params['variance']}",
    ]

    print(f"Running command: {' '.join(cmd)}")
    start_time = time.time()
    process = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()

    duration = end_time - start_time

    # 生成更规范的日志文件名（处理小数点）
    param_str = f"mean_{str(params['mean']).replace('.','p')}_var_{str(params['variance']).replace('.','p')}"
    log_file = log_dir / f"noise_experiment_{param_str}.log"

    # 增强日志记录
    log_data = {
        'experiment_type': 'noise_parameter_study',
        'parameters': params,
        'timing': {
            'start': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)),
            'end': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)),
            'duration_seconds': round(duration, 2)
        },
        'system_info': {
            'cpu_cores': os.cpu_count(),
            'gpu_available': torch.cuda.is_available()
        },
        'exit_status': process.returncode,
        'error_log': process.stderr if process.returncode != 0 else None
    }

    with open(log_file, 'w') as f:
        yaml.dump(log_data, f, sort_keys=False)

    if process.returncode == 0:
        print(f"Experiment {param_str} completed successfully.")
    else:
        print(f"Experiment {param_str} failed with error:\n{process.stderr}")

def main():
    # 定义噪声参数空间
    noise_params = {
        'mean': [0, 1, 10],      # 噪声均值
        'variance': [0.1, 1.0, 10.0, 100.0],  # 噪声方差
    }

    # 创建带时间戳的日志目录
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_dir = Path(f'experiment_logs/noise_study_{timestamp}')
    log_dir.mkdir(parents=True, exist_ok=True)

    # 生成所有参数组合
    param_combinations = [
        dict(zip(noise_params.keys(), values))
        for values in itertools.product(*noise_params.values())
    ]

    print(f"Total experiments to run: {len(param_combinations)}")

    # 运行所有实验
    total_start = time.time()
    for idx, params in enumerate(param_combinations, 1):
        print(f"\n=== Running experiment {idx}/{len(param_combinations)} ===")
        run_noise_experiment(params, log_dir)
    
    # 生成汇总报告
    total_duration = time.time() - total_start
    summary = {
        'total_experiments': len(param_combinations),
        'successful': sum(1 for f in log_dir.glob('*.log') if 'error' not in yaml.safe_load(open(f))),
        'total_duration_hours': round(total_duration / 3600, 2),
        'average_duration_per_experiment': round(total_duration / len(param_combinations), 2),
        'parameter_space': noise_params
    }

    with open(log_dir / 'summary.yaml', 'w') as f:
        yaml.dump(summary, f, sort_keys=False)

    print(f"\nExperiment batch completed. Summary saved in {log_dir}/summary.yaml")

if __name__ == '__main__':
    main() 