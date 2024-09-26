

import torch
import os
from collections import OrderedDict


def deepspeed_unwrapper(checkpoint_dir, output_dir):
    state_dict = torch.load(os.path.join(checkpoint_dir), map_location='cpu')['module']

    unet_state = OrderedDict()
    controlnet_state = OrderedDict()
    for name, data in state_dict.items():
        model = name.split('.', 1)[0]
        module = name.split('.', 1)[1]
        if model == 'unet':
            unet_state[module] = data
        elif model == 'controlnet':
            controlnet_state[module] = data
    
    for model in ['unet', 'controlnet']:
        if not os.path.exists(os.path.join(output_dir, model)):
            os.makedirs(os.path.join(output_dir, model))


    torch.save(unet_state, os.path.join(output_dir, "unet", "diffusion_pytorch_model.bin"))
    torch.save(controlnet_state, os.path.join(output_dir, "controlnet", "diffusion_pytorch_model.bin"))