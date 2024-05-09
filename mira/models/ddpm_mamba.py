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

from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin
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



class MambaPlugin(pl.LightningModule, ModelMixin, ConfigMixin):
    def __init__(self, model_num=28):
        super().__init__()
        self.init_model(
            1152,
            model_num
        )
    
    def init_model(self, model_dim, model_num):
        self.mamba_list = [Mamba(
            d_model = model_dim
        ) for _ in range(model_num)]
        for i, each in enumerate(self.mamba_list):
            self.add_module(f'mamba_{i}', each)

class MiraMamba(MiraDDPM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decoder_clip_length = 72
        self.mamba_plugin = MambaPlugin()
        self.inject_decoder_forward()
        self.inject_mamba()

    def inject_decoder_forward(self):
        decoder = self.first_stage_model.decoder
        def new_forward(z, **kwargs):
            video_length = z.shape[0]
            output = []
            for i in range(0, video_length, self.decoder_clip_length):
                z_clip = z[i:i+self.decoder_clip_length]
                z_clip = decoder.old_forward(
                    z_clip, 
                    **{
                        **kwargs,
                        'timesteps': z_clip.shape[0]
                    }
                )
                output.append(z_clip)
            output = torch.cat(output, dim=0)
            return output
        if not hasattr(decoder, 'old_forward'):
            decoder.old_forward = decoder.forward
        decoder.forward = new_forward

    def get_attn_newforward(self, module, module_idx):
        def new_forward(x, encoder_hidden_states=None, attention_mask=None):
            context, mask = encoder_hidden_states, attention_mask
            temporal_n = x.shape[1]
            mamba_module = self.mamba_plugin.mamba_list[module_idx]

            # return mamba_module(x) + mamba_module(x.flip(1))

            assert not temporal_n % 60
            q = module.to_q(x)
            
            mode = 'slide_window'

            forward_context = mamba_module(x)
            backward_context = mamba_module(x.flip(1))
            context = x + forward_context + backward_context
            # context = forward_context + backward_context
            # context = x if context is None else context


            k, v = module.to_k(context), module.to_v(context)
            b, _, _ = q.shape
            q, k, v = map(
                lambda t: t.unsqueeze(3).reshape(b, t.shape[1], module.heads, -1).permute(0, 2, 1, 3).reshape(b*module.heads, t.shape[1], -1),
                (q, k, v),
            )

            # independtly
            if mode == "independtly":
                out = []
                for i in range(0, temporal_n, 60):
                    out.append(
                        torch.nn.functional.scaled_dot_product_attention(
                            q[:, i:i+60], k[:, i:i+60], v[:, i:i+60],
                            attn_mask=None, dropout_p=0.0, is_causal=False
                        )
                    )
                out = torch.cat(out, dim=1)
            elif mode == "slide_window":
                #  padding k and v to the head and tail
                padding = torch.zeros_like(q[:,:30])
                q = torch.cat([padding, q, padding], 1)
                v = torch.cat([padding, v, padding], 1)

                # 生成一个mask
                mask = torch.ones(60, 120).to(q.device)
                for i in range(60):
                    mask[i, : i] = 0
                    mask[i, 60 - i:] = 0
                
                out = []
                for i in range(0, temporal_n, 60):
                    out.append(
                        torch.nn.functional.scaled_dot_product_attention(
                            q[:, i:i+60], k[:, i:i+120], v[:, i:i+120],
                            attn_mask=mask, dropout_p=0.0, is_causal=False
                        )
                    )
                out = torch.cat(out, dim=1)


            out = (
                out.unsqueeze(0).reshape(b, module.heads, out.shape[1], -1).permute(0, 2, 1, 3)
                .reshape(b, out.shape[1], -1)
            )

            # linear proj
            hidden_states = module.to_out[0](out)
            hidden_states = module.to_out[1](hidden_states)
            
            return hidden_states
        return new_forward

    def inject_mamba(self):
        attention_module = [
            each.attn1 for each in 
            self.model.diffusion_model.temporal_transformer_blocks
        ]
        for i, each in enumerate(attention_module):
            print(f'Injecting mamba to module {i}')
            if not hasattr(each,'old_forward'):
                each.old_forward = each.forward
            each.forward = self.get_attn_newforward(each, i)



    def configure_optimizers(self):

        """ configure_optimizers for LatentDiffusion """
        lr = self.learning_rate

        params = list(self.model.parameters()) + list(self.mamba_plugin.parameters())
        mainlogger.info(f"@Training [{len(params)}] Full Paramters.")

        # params = list(self.mamba_plugin.parameters())
        # mainlogger.info(f"@Training [{len(params)}] Mamva Paramters.")

        if self.learn_logvar:
            mainlogger.info('Diffusion model optimizing logvar')
            if isinstance(params[0], dict):
                params.append({"params": [self.logvar]})
            else:
                params.append(self.logvar)
        optimizer = torch.optim.AdamW(params, lr=lr)


        ## lr scheduler
        if self.use_scheduler:
            mainlogger.info("Setting up LambdaLR scheduler...")
            lr_scheduler = self.configure_schedulers(optimizer)
            return [optimizer], [lr_scheduler]

        return optimizer