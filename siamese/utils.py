import random
import numpy as np
import torch
import pandas as pd
import os
import wandb

def seed_everything(seed):
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set PyTorch random seed for CPU and GPU
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Set PyTorch deterministic operations for cudnn backend
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def wandb_settings(key, config, project, entity, name):

    # Wandb Login
    wandb.login(key=key)

    # Initialize W&B
    run = wandb.init(
        config=config,
        project=project,
        entity=entity,
        name=name,
    )