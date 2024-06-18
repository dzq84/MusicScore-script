import os
from omegaconf import OmegaConf
import numpy as np
import torch
from torchvision.models import inception_v3
from scipy.linalg import sqrtm
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage

from datetime import datetime
from rich import print
from tqdm import tqdm, trange
import argparse

from ddim import DDIMSampler
from dataset import HDToy, HDData14k, HDData200k
from utils import count_params, load_model_from_config, instantiate_from_config

device = torch.device("cuda")

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str)
parser.add_argument("--scale", type=str)

args = parser.parse_args()

config = OmegaConf.load("./score_v0.1_train_unet.yaml")
print(f"Loading model from ./score_v0.1_train_unet.yaml")
ckpt_path = "../unet_finetune/0525_toy_global_0_lr_1e-5_no_sample_distribution/last.pt"
pl_sd = torch.load(ckpt_path, map_location="cpu")

if pl_sd.get("model_state_dict") is not None:
    state_dict = pl_sd["model_state_dict"]
    if pl_sd.get("global_step") is not None:
        global_step = pl_sd["global_step"]
else:
    state_dict = pl_sd
model = instantiate_from_config(config.model)

print(f"Start loading Stable Diffusion...")
m, u = model.load_state_dict(state_dict, strict=False)
verbose = True
if len(m) > 0 and verbose:
    print("missing keys:")
    print(m)
if len(u) > 0 and verbose:
    print("unexpected keys:")
    print(u)
print(f"Finish loading Stable Diffusion")

count_params(model, verbose=True)

model.to(device)

batch_size = 1

if args.scale == "MS-400":
    real_dataset = HDToy(instance_data_root=args.data_dir, size=512)
elif args.scale == "MS-14k":
    real_dataset = HDData14k(instance_data_root=args.data_dir, size=512)
elif args.scale == "MS-200k":
    real_dataset = HDData200k(instance_data_root=args.data_dir, size=512)

train_dataloader = DataLoader(
    real_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=14,
)

C = 4
H, W = 512, 512
resolution = 512
f = 8
shape = [C, H // f, W // f]
ddim_steps = 250
sampler = DDIMSampler(model, device=device)
print(
    f"""
DDIM Sampler Configuration:
C: {C}
H: {H} W: {W}
k: {f}
shape: {shape}
ddim_steps: {ddim_steps}
batch_size: {batch_size}
dataset size: {len(real_dataset)}
time: {datetime.now()}
global_step: {global_step}
"""
)

from tqdm import tqdm

model.eval()
with torch.no_grad(), model.ema_scope():
    for batch in tqdm(train_dataloader):
        text = batch["caption"]
        file_name = batch["image_path"]
        images = batch["image"].to(device)

        c = model.get_learned_conditioning(text)
        uc = model.get_learned_conditioning(len(text) * [""])

        z_denoise, intermediates = sampler.sample(
            S=ddim_steps,
            batch_size=batch_size,
            shape=shape,
            conditioning=c,
            verbose=False,
            log_every_t=500,
            unconditional_guidance_scale=4,
            unconditional_conditioning=uc,
        )

        output = model.decode_first_stage(z_denoise)

        output = torch.clamp((output + 1.0) / 2.0, min=0.0, max=1.0)

        img = ToPILImage(mode="RGB")(output[0])
        out_path = args.data_dir.replace("real_images", "generated_images") + file_name[0]
        img.save(out_path)
