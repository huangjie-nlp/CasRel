from framework.framework import Framework
from config.config import Config
import torch
import random
import numpy as np

seed = 2022

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

config = Config()
fw = Framework(config)
fw.train()
print("===================================test=============================")
fw.test(config.test_data)
