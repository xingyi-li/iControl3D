import gc
import io
import math
import sys

from PIL import Image, ImageOps, ImageDraw
import requests
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.ops import masks_to_boxes
from torch import autocast
import contextlib
from tqdm import tqdm

import numpy as np

from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults, classifier_defaults, create_classifier

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config

from einops import rearrange
from math import log2, sqrt

import argparse
import pickle

import os

import skimage
import skimage.measure

from transformers import CLIPTokenizer, CLIPTextModel

import tempfile
from urllib.request import urlopen, Request
import shutil
from pathlib import Path

try:
    SAMPLING_MODE = Image.Resampling.LANCZOS
except Exception as e:
    SAMPLING_MODE = Image.LANCZOS

# argument parsing
def download_url_to_file(url, dst, hash_prefix=None, progress=True):
    r"""Download object at the given URL to a local path.

    Args:
        url (string): URL of the object to download
        dst (string): Full path where object will be saved, e.g. ``/tmp/temporary_file``
        hash_prefix (string, optional): If not None, the SHA256 downloaded file should start with ``hash_prefix``.
            Default: None
        progress (bool, optional): whether or not to display a progress bar to stderr
            Default: True
    https://pytorch.org/docs/stable/_modules/torch/hub.html#load_state_dict_from_url
    """
    file_size = None
    req = Request(url)
    u = urlopen(req)
    meta = u.info()
    if hasattr(meta, 'getheaders'):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])

    # We deliberately save it in a temp file and move it after
    # download is complete. This prevents a local working checkpoint
    # being overridden by a broken download.
    dst = os.path.expanduser(dst)
    dst_dir = os.path.dirname(dst)
    f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)

    try:
        with tqdm(total=file_size, disable=not progress,
                  unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                pbar.update(len(buffer))

        f.close()
        shutil.move(f.name, dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)

def fetch(url_or_path):
    if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, 'rb')

class GlidModel:
    def __init__(self,config) -> None:
        self.get_parser()
        self.args = self.parser.parse_args(config)
        self.fetch_weights()
        self.setup_model()
    
    def fetch_weights(self):
        cwd=os.getcwd()
        if not os.path.exists("kl.pt"):
            download_url_to_file(url="https://huggingface.co/Jack000/glid-3-xl-stable/resolve/main/default/kl-1.4.pt",dst=os.path.join(cwd,"kl.pt"))
        if not os.path.exists("inpaint.pt"):
            download_url_to_file(url="https://huggingface.co/Jack000/glid-3-xl-stable/resolve/main/inpaint/ema_0.9999_100000.pt",dst=os.path.join(cwd,"inpaint.pt"))

    def get_parser(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('--model_path', type=str, default = 'inpaint.pt',
                   help='path to the diffusion model')

        parser.add_argument('--kl_path', type=str, default = 'kl.pt',
                   help='path to the LDM first stage model')

        parser.add_argument('--text', type = str, required = False, default = '',
                    help='your text prompt')

        parser.add_argument('--classifier', type=str, default = '',
                   help='path to the classifier model')

        parser.add_argument('--classifier_scale', type = int, required = False, default = 100,
                    help='amount of classifier guidance')

        parser.add_argument('--edit', type = str, required = False,
                    help='path to the image you want to edit (either an image file or .npy containing a numpy array of the image embeddings)')

        parser.add_argument('--outpaint', type = str, required = False, default = '',
                    help='options: expand (all directions), wider, taller, left, right, top, bottom')

        parser.add_argument('--mask', type = str, required = False,
                    help='path to a mask image. white pixels = keep, black pixels = discard. width = image width/8, height = image height/8')

        parser.add_argument('--negative', type = str, required = False, default = '',
                    help='negative text prompt')

        parser.add_argument('--init_image', type=str, required = False, default = None,
                   help='init image to use')

        parser.add_argument('--skip_timesteps', type=int, required = False, default = 0,
                   help='how many diffusion steps are gonna be skipped')

        parser.add_argument('--prefix', type = str, required = False, default = '',
                    help='prefix for output files')

        parser.add_argument('--num_batches', type = int, default = 1, required = False,
                    help='number of batches')

        parser.add_argument('--batch_size', type = int, default = 1, required = False,
                    help='batch size')

        parser.add_argument('--width', type = int, default = 512, required = False,
                    help='image size of output (multiple of 8)')

        parser.add_argument('--height', type = int, default = 512, required = False,
                    help='image size of output (multiple of 8)')

        parser.add_argument('--seed', type = int, default=-1, required = False,
                    help='random seed')

        parser.add_argument('--guidance_scale', type = float, default = 7.0, required = False,
                    help='classifier-free guidance scale')

        parser.add_argument('--steps', type = int, default = 0, required = False,
                    help='number of diffusion steps')

        parser.add_argument('--cpu', dest='cpu', action='store_true')

        parser.add_argument('--ddim', dest='ddim', action='store_true')

        parser.add_argument('--ddpm', dest='ddpm', action='store_true')
        self.parser=parser

    def setup_model(self):
        args=self.args
        device = torch.device('cuda' if (torch.cuda.is_available() and not args.cpu) else 'cpu')
        # print('Using device:', device)
        self.device=device
        
        model_state_dict = torch.load(args.model_path, map_location='cpu')
        
        model_params = {
            'attention_resolutions': '32,16,8',
            'class_cond': False,
            'diffusion_steps': 1000,
            'rescale_timesteps': True,
            'timestep_respacing': '50',
            'image_size': 32,
            'learn_sigma': False,
            'noise_schedule': 'linear',
            'num_channels': 320,
            'num_heads': 8,
            'num_res_blocks': 2,
            'resblock_updown': False,
            'use_fp16': True,
            'use_scale_shift_norm': False,
            'clip_embed_dim': None,
            'image_condition': True if model_state_dict['input_blocks.0.0.weight'].shape[1] == 8 else False,
            'super_res_condition': True if 'external_block.0.0.weight' in model_state_dict else False,
        }
        self.model_params=model_params
        if args.ddpm:
            model_params['timestep_respacing'] = '1000'
        if args.ddim:
            if args.steps:
                model_params['timestep_respacing'] = 'ddim'+str(args.steps)
            else:
                model_params['timestep_respacing'] = 'ddim250'
        elif args.steps:
            model_params['timestep_respacing'] = str(args.steps)
        
        model_config = model_and_diffusion_defaults()
        model_config.update(model_params)
        self.model_config=model_config
        if args.cpu:
            model_config['use_fp16'] = False
            autocast = contextlib.nullcontext
        
        # Load models
        model, diffusion = create_model_and_diffusion(**model_config)
        model.load_state_dict(model_state_dict, strict=True)
        model.requires_grad_(False).eval().to(device)
        
        if model_config['use_fp16']:
            model.convert_to_fp16()
        else:
            model.convert_to_fp32()
        
        def set_requires_grad(model, value):
            for param in model.parameters():
                param.requires_grad = value
        
        # load classifier
        if args.classifier:
            classifier_config = classifier_defaults()
            classifier_config['classifier_width'] = 128
            classifier_config['classifier_depth'] = 4
            classifier_config['classifier_attention_resolutions'] = '64,32,16,8'
            classifier = create_classifier(**classifier_config)
            classifier.load_state_dict(
                torch.load(args.classifier, map_location="cpu")
            )
            classifier.to(device)
            classifier.convert_to_fp16()
            classifier.eval()
            self.classifier=classifier
        
        # vae
        kl_config = OmegaConf.load(os.path.join(Path(__file__).parent,'kl.yaml'))
        kl_sd = torch.load(args.kl_path, map_location="cpu")
        
        ldm = instantiate_from_config(kl_config.model)
        ldm.load_state_dict(kl_sd, strict=True)
        
        ldm.to(device)
        ldm.eval()
        ldm.requires_grad_(False)
        set_requires_grad(ldm, False)
        
        # clip
        clip_version = 'openai/clip-vit-large-patch14'
        clip_tokenizer = CLIPTokenizer.from_pretrained(clip_version)
        clip_transformer = CLIPTextModel.from_pretrained(clip_version)
        clip_transformer.eval().requires_grad_(False).to(device)
        self.model=model
        self.clip_tokenizer=clip_tokenizer
        self.clip_transformer=clip_transformer
        self.ldm=ldm
        self.diffusion=diffusion
    
    def run(self,image_pil,prompt,guidance_scale=7.0,negative_prompt="",seed=-1,height=512,width=512,**kwargs):
        gc.collect()
        args=self.args
        args.width=width
        args.height=height
        args.batch_size=kwargs["generate_num"]
        if kwargs["use_seed"]:
            torch.manual_seed(seed)
        device=self.device
        # clip context
        text = self.clip_tokenizer([prompt]*args.batch_size, truncation=True, max_length=77, return_length=True, return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        text_blank = self.clip_tokenizer([negative_prompt]*args.batch_size, truncation=True, max_length=77, return_length=True, return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        text_tokens = text["input_ids"].to(device)
        text_blank_tokens = text_blank["input_ids"].to(device)

        text_emb = self.clip_transformer(input_ids=text_tokens).last_hidden_state
        text_emb_blank = self.clip_transformer(input_ids=text_blank_tokens).last_hidden_state

        image_embed = None
        image_alpha_pil = image_pil.resize(
            (width,height), resample=SAMPLING_MODE,
        )
        sel_buffer = np.array(image_alpha_pil)
        img_arr = sel_buffer[:, :, 0:3]
        mask_arr = sel_buffer[:, :, -1]
        # mask_arr = skimage.measure.block_reduce(mask_arr, (8, 8), np.min)
        # mask_arr = mask_arr.repeat(8, axis=0).repeat(8, axis=1)
        # mask_arr = mask_arr[:,:,np.newaxis].repeat(3,axis=2)
        # img_arr[mask_arr<10]=0
        # image context
        if True:
            input_image_pil = Image.fromarray(img_arr)

            im = transforms.ToTensor()(input_image_pil).unsqueeze(0).to(device)
            im = 2*im-1
            im = self.ldm.encode(im).sample()

            input_image = im
            input_image_mask = torch.ones(1,1,im.shape[2], im.shape[3], device=device, dtype=torch.bool)

            input_image_pil = self.ldm.decode(input_image)
            input_image_pil = TF.to_pil_image(input_image_pil.squeeze(0).add(1).div(2).clamp(0, 1))

            input_image *= 0.18215

            if args.mask:

                mask_image = Image.fromarray(mask_arr).convert('L')
                mask_image = mask_image.resize((input_image.shape[3],input_image.shape[2]), resample=SAMPLING_MODE)
                mask = transforms.ToTensor()(mask_image).unsqueeze(0).to(device)

            mask1 = (mask > 0.5)
            input_image_mask *= mask1

            #mask1 = mask1.float()
            #input_image *= mask1


            image_embed = torch.cat(args.batch_size*2*[input_image], dim=0).float()

        else:
            # using inpaint model but no image is provided
            image_embed = torch.zeros(args.batch_size*2, 4, args.height//8, args.width//8, device=device)

        kwargs = {
            "context": torch.cat([text_emb, text_emb_blank], dim=0).float(),
            "clip_embed": None,
            "image_embed": image_embed
        }

        # Create a classifier-free guidance sampling function
        def model_fn(x_t, ts, **kwargs):
            half = x_t[: len(x_t) // 2]
            combined = torch.cat([half, half], dim=0)
            model_out = self.model(combined, ts, **kwargs)
            eps, rest = model_out[:, :3], model_out[:, 3:]
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            return torch.cat([eps, rest], dim=1)

        cond_fn = None

        if args.classifier:
            def cond_fn(x, t, context=None, clip_embed=None, image_embed=None):
                with torch.enable_grad():
                    x_in = x[:x.shape[0]//2].detach().requires_grad_(True)
                    logits = self.classifier(x_in, t)
                    log_probs = F.log_softmax(logits, dim=-1)
                    selected = log_probs[range(len(logits)), torch.ones(x_in.shape[0], dtype=torch.long)]
                    return torch.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale

        cur_t = None
    
        if args.ddpm:
            sample_fn = self.diffusion.ddpm_sample_loop_progressive
        elif args.ddim:
            sample_fn = self.diffusion.ddim_sample_loop_progressive
        else:
            sample_fn = self.diffusion.plms_sample_loop_progressive

        def save_sample(i, samples, square=None):
            lst=[]
            for k, image in enumerate(samples):
                image_scaled = image/0.18215
                im = image_scaled.unsqueeze(0)
                out = self.ldm.decode(im)

                out = TF.to_pil_image(out.squeeze(0).add(1).div(2).clamp(0, 1))
                lst.append(out)
            return lst

        init = None
        overlap = 32
        with autocast("cuda"):
            ret=[]
            for i in range(args.num_batches):
                output = input_image.detach().clone()
                output *= input_image_mask.repeat(1, 4, 1, 1).float()

                mask = input_image_mask.detach().clone()

                box = masks_to_boxes(~mask.squeeze(0))[0]

                x0 = int(box[0])
                y0 = int(box[1])
                x1 = int(box[2] + 1)
                y1 = int(box[3] + 1)

                x_num = math.ceil(((x1-x0)-overlap)/(64-overlap))
                y_num = math.ceil(((y1-y0)-overlap)/(64-overlap))

                if x_num < 1:
                    x_num = 1
                if y_num < 1:
                    y_num = 1

                for y in range(y_num):
                    for x in range(x_num):
                        offsetx = x0 + x*(64-overlap)
                        offsety = y0 + y*(64-overlap)

                        if offsetx + 64 > x1:
                            offsetx = x1 - 64
                        if offsetx < 0:
                            offsetx = 0

                        if offsety + 64 > y1:
                            offsety = y1 - 64
                        if offsety < 0:
                            offsety = 0

                        patch_input = output[:,:, offsety:offsety+64, offsetx:offsetx+64]
                        patch_mask = mask[:,:, offsety:offsety+64, offsetx:offsetx+64]

                        if not torch.any(~patch_mask):
                            # region does not require any inpainting
                            output[:,:, offsety:offsety+64, offsetx:offsetx+64] = patch_input
                            continue

                        mask[:,:, offsety:offsety+64, offsetx:offsetx+64] = True

                        patch_init = None

                        if args.skip_timesteps > 0:
                            patch_init = input_image[:,:, offsety:offsety+64, offsetx:offsetx+64]
                            patch_init = torch.cat([patch_init, patch_init], dim=0)

                        skip_timesteps = args.skip_timesteps

                        if not torch.any(patch_mask):
                            # region has no input image, cannot use init
                            patch_init = None
                            skip_timesteps = 0

                        patch_kwargs = {
                            "context": kwargs["context"],
                            "clip_embed": None,
                            "image_embed": torch.cat([patch_input, patch_input], dim=0)
                        }

                        cur_t = self.diffusion.num_timesteps - 1

                        samples = sample_fn(
                            model_fn,
                            (2, 4, 64, 64),
                            clip_denoised=False,
                            model_kwargs=patch_kwargs,
                            cond_fn=cond_fn,
                            device=device,
                            progress=True,
                            init_image=patch_init,
                            skip_timesteps=skip_timesteps,
                        )

                        for j, sample in enumerate(samples):
                            cur_t -= 1
                            output[0,:, offsety:offsety+64, offsetx:offsetx+64] = sample['pred_xstart'][0]
                            # if j % 25 == 0:
                                # save_sample(i, output, square=(offsetx, offsety))

                        ret.append(save_sample(i, output))
        return ret[0]



