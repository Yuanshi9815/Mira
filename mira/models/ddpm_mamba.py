import os, random
from functools import partial
import numpy as np
from tqdm import tqdm
from einops import rearrange, repeat
import logging

mainlogger = logging.getLogger('mainlogger')


import torch
from torchvision.utils import make_grid
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import torch.distributed as dist
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

from utils.utils import instantiate_from_config, count_params, check_istarget
from mira.distributions import normal_kl, DiagonalGaussianDistribution
from mira.models.utils_diffusion import make_beta_schedule
from mira.models.samplers.ddim import DDIMSampler
from mira.basics import disabled_train
from mira.common import (
    extract_into_tensor,
    noise_like,
    exists,
    default
)
import itertools
from pytorch_lightning import seed_everything
from utils.save_video import npz_to_video_grid
from mira.scripts.evaluation.inference import inference_prompt, load_prompts

from torchvision import transforms

__conditioning_keys__ = {'concat': 'c_concat',
                         'crossattn': 'c_crossattn',
                         'adm': 'y'}

from mira.models.base_ddpm import DDPM
from mira.models.ddpm3d import MiraDDPM
from mamba_ssm import Mamba



class MambaPlugin(pl.LightningModule):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_model(
            1152,
            27
        )
    
    def init_model(self, model_dim, model_num):
        self.mamba_list = [Mamba(
            d_model = model_dim
        ) for _ in range(model_num)]

class MiraMamba(MiraDDPM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mamba_plugin = MambaPlugin()

    def inject_mamba(self, config):
        self.mamba = True
        self.mamba_config = config

    