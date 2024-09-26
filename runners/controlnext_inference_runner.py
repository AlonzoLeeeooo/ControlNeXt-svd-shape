import os
import torch
import numpy as np
from PIL import Image
from decord import VideoReader
from transformers import CLIPVisionModelWithProjection
from pipeline.pipeline_stable_video_diffusion_controlnext import StableVideoDiffusionPipelineControlNeXt
from models.controlnext_vid_svd import ControlNeXtSVDModel
from models.unet_spatio_temporal_condition_controlnext import UNetSpatioTemporalConditionControlNeXtModel
from diffusers import AutoencoderKLTemporalDecoder
from utils.data_utils import load_tensor, save_vid_side_by_side

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def controlnext_inference_runner(args):
    assert (args.validation_control_images_folder is None) ^ (args.validation_control_video_path is None), "must and only one of [validation_control_images_folder, validation_control_video_path] should be given"

    unet = UNetSpatioTemporalConditionControlNeXtModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        low_cpu_mem_usage=True,
    )
    controlnext = ControlNeXtSVDModel()
    controlnext_state_dict = load_tensor(args.controlnext_path)
    controlnext.load_state_dict(controlnext_state_dict)
    unet.load_state_dict(load_tensor(args.unet_path), strict=False)

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="image_encoder")
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae")
    
    pipeline = StableVideoDiffusionPipelineControlNeXt.from_pretrained(
        args.pretrained_model_name_or_path,
        controlnext=controlnext, 
        unet=unet,
        vae=vae,
        image_encoder=image_encoder)
    pipeline.to(device)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load the reference image
    ref_image = Image.open(args.ref_image_path).convert('RGB')
    ref_image = ref_image.resize((args.width, args.height))
    
    # Load the control images
    validation_control_images = []
    validation_control_images = [Image.open(os.path.join(args.validation_control_images_folder, img)) for img in sorted(os.listdir(args.validation_control_images_folder))]
    validation_control_images = [img.resize((args.width, args.height)) for img in validation_control_images]


    
    final_result = []
    frames = args.batch_frames
    num_frames = min(args.max_frame_num, len(validation_control_images)) 

    for i in range(num_frames):
        validation_control_images[i] = Image.fromarray(np.array(validation_control_images[i]))
    
    # Set up PyTorch generator for reproducibility
    generator = torch.Generator(device=device).manual_seed(args.seed)

    video_frames = pipeline(
        ref_image, 
        validation_control_images[:num_frames], 
        decode_chunk_size=2,
        num_frames=num_frames,
        motion_bucket_id=127.0, 
        fps=7,
        controlnext_cond_scale=1.0, 
        width=args.width, 
        height=args.height, 
        min_guidance_scale=args.guidance_scale, 
        max_guidance_scale=args.guidance_scale, 
        frames_per_batch=frames, 
        num_inference_steps=args.num_inference_steps, 
        overlap=args.overlap,
        device=device,
        generator=generator).frames[0]
    final_result.append(video_frames)

    if device == "mps":
        fps = 16
    else:
        fps = VideoReader(args.validation_control_video_path).get_avg_fps()  // args.sample_stride
    
    filename = os.path.basename(args.validation_control_images_folder).split(".")[0]
    save_vid_side_by_side(
        final_result, 
        validation_control_images[:num_frames], 
        args.output_dir, 
        fps=fps,
        filename=filename,
        seed=args.seed)