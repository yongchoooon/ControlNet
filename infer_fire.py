from share import *
import config

import cv2
import einops
import numpy as np
import torch
import random
import os
import json

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.midas import MidasDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from PIL import Image
from tqdm import tqdm



apply_midas = MidasDetector()

model = create_model('./models/cldm_v21.yaml').cpu()
model.load_state_dict(load_state_dict('./lightning_logs/version_1/checkpoints/epoch=45-step=205620.ckpt', location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)


def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, bg_threshold):
    with torch.no_grad():
        input_image = HWC3(input_image) # 이미지 전처리
        
       # _, detected_map = apply_midas(resize_image(input_image, detect_resolution), bg_th=bg_threshold)
       # detected_map = HWC3(input_image)  
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

       # detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR) 
        input_image = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LINEAR) 
    
    #   input_image = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR) 

       # control = torch.from_numpy(detected_map[:, :, ::-1].copy()).float().cuda() / 255.0
        control = torch.from_numpy(input_image[:, :, ::-1].copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return results

def load_image_to_ndarray(file_path):
    try:
        pil_image = Image.open(file_path)
        image_array = np.array(pil_image)
        return image_array
    except IOError:
        print(f"Unable to load the image from file: {file_path}")
        return None
    
# Parameters
a_prompt = 'best quality, extremely detailed'
n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
num_samples = 1
image_resolution = 512
detect_resolution = 384
ddim_steps = 50
guess_mode = False
strength = 1.0
scale = 9.0
eta = 0.0
bg_threshold = 0.4

# Inputs
relative_path = '../fire_gen/fire_and_masked_images_30_real_new/train/masked/'
image_file_list = sorted([relative_path + i for i in os.listdir(relative_path)])
prompt_json_path = '../fire_gen/fire_and_masked_images_30_real_new/train/BLIP+chatgpt_prompt_real_new_train_lora_cleaned_controlnet.json'
with open(prompt_json_path, 'r') as f:
    prompt_json_data = json.load(f)
    f.close()
prompt_list = [j['best_n'] for j in prompt_json_data]

output_dir = 'inference_fire'
os.makedirs(output_dir, exist_ok = True)

for i in tqdm(range(len(image_file_list))):
    seed = random.randint(1000000, 9999999)
    image_file_path = image_file_list[i]
    input_image = load_image_to_ndarray(image_file_path)
    prompt = prompt_list[i]

    infer = process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, bg_threshold)

    pil_image = Image.fromarray(infer[0])

    image_file_name = image_file_path
    pil_image.save(os.path.join(output_dir, image_file_path[-19:]))