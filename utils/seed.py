"""
随机种子设置工具
确保实验可复现性
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    设置全局随机种子，确保实验可复现
    
    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 确保卷积运算的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 设置环境变量
    os.environ['PYTHONHASHSEED'] = str(seed)


def worker_init_fn(worker_id: int) -> None:
    """
    DataLoader worker 初始化函数
    确保多进程数据加载的可复现性
    
    Args:
        worker_id: worker进程ID
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

