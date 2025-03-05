import subprocess
from pathlib import Path
import itertools
import time
import yaml

def run_experiment(params, log_dir):
    """运行单次实验并记录时间"""
    cmd = [
        'python',
        'src/main.py',
        f"distributed.num_agents={params['num_agents']}",
        f"env.train_steps={params['train_steps']}",
        f"env.task={params['task']}"
    ]

    print(f"Running command: {' '.join(cmd)}")
    start_time = time.time()
    process = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()

    duration = end_time - start_time

    # 保存实验结果到日志
    log_file = log_dir / f"experiment_{params['task']}_agents_{params['num_agents']}_steps_{params['train_steps']}.yaml-change-log"
    log_data = {
        'parameters': params,
        'start_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)),
        'end_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)),
        'duration_seconds': duration,
        'status': 'success' if process.returncode == 0 else 'failure',
        'stderr': process.stderr if process.returncode != 0 else None
    }

    with open(log_file, 'w') as f:
        yaml.dump(log_data, f)

    if process.returncode == 0:
        print(f"Experiment with parameters {params} completed successfully.")
    else:
        print(f"Experiment with parameters {params} failed.")
        print(process.stderr)

def main():

    # 实验参数组合
    experiment_params = {
        'num_agents': [1,4,7,11],
        'train_steps': [500000],
        'task': ['mw-door-unlock']
    }

    # 创建日志目录
    log_dir = Path('experiment_logs')
    log_dir.mkdir(parents=True, exist_ok=True)

    # 生成所有参数组合
    param_combinations = [dict(zip(experiment_params.keys(), v)) 
                         for v in itertools.product(*experiment_params.values())]

    # 运行实验并记录总时间
    total_start_time = time.time()

    for params in param_combinations:
        print("\nStarting experiment with parameters:", params)
        run_experiment(params, log_dir)

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time

    # 保存总体实验时间
    summary_file = log_dir / "summary.yaml"
    summary_data = {
        'total_experiments': len(param_combinations),
        'total_duration_seconds': total_duration,
        'finished_at': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(total_end_time))
    }

    with open(summary_file, 'w') as f:
        yaml.dump(summary_data, f)

    print(f"All experiments completed. Total duration: {total_duration / 3600:.2f} hours")

if __name__ == '__main__':
    main()
