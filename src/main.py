# from train import Trainer
# from copy import deepcopy
# import hydra
# from omegaconf import DictConfig

# @hydra.main(config_name='config', config_path='.')
# def main(cfg: DictConfig):
#     trainer = Trainer(cfg)
#     trainer.run()


# if __name__ == "__main__":
#     main()

import hydra
import torch
from distributed.trainer import DistributedTrainer

@hydra.main(config_path=".", config_name="config")
def main(cfg):
    # 使用分布式训练器替换原有训练器
    trainer = DistributedTrainer(cfg)
    trainer.run()

if __name__ == "__main__":
    main()