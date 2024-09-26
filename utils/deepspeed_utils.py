import os
import torch
import torch.nn as nn
from collections import OrderedDict

class DeepSpeedWrapperModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        for name, value in kwargs.items():
            assert isinstance(value, nn.Module)
            self.register_module(name, value)





def deepspeed_unwrapper(checkpoint_dir, output_dir):
    state_dict = torch.load(os.path.join(checkpoint_dir), map_location='cpu')['module']

    unet_state = OrderedDict()
    controlnet_state = OrderedDict()
    for name, data in state_dict.items():
        if "unet" in name:
            unet_state[name.replace("unet.", "")] = data
        elif "controlnext" in name:
            controlnet_state[name.replace("controlnext.", "")] = data
    
    for model in ['unet', 'controlnext']:
        if not os.path.exists(os.path.join(output_dir, model)):
            os.makedirs(os.path.join(output_dir, model))


    torch.save(unet_state, os.path.join(output_dir, "unet", "diffusion_pytorch_model.bin"))
    torch.save(controlnet_state, os.path.join(output_dir, "controlnext", "diffusion_pytorch_model.bin"))