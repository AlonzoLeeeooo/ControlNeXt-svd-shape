import os
import argparse
from utils.deepspeed_utils import deepspeed_unwrapper
from runners.controlnext_inference_runner import controlnext_inference_runner


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train Stable Diffusion XL for InstructPix2Pix."
    )

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True
    )

    parser.add_argument(
        "--validation_control_images_folder",
        type=str,
        default=None,
        required=False,
    )

    parser.add_argument(
        "--validation_control_video_path",
        type=str,
        default=None,
        required=False,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        required=True
    )

    parser.add_argument(
        "--height",
        type=int,
        default=768,
        required=False
    )

    parser.add_argument(
        "--width",
        type=int,
        default=512,
        required=False
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=2.,
        required=False
    )

    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=25,
        required=False
    )


    parser.add_argument(
        "--controlnext_path",
        type=str,
        default=None,
        required=False
    )

    parser.add_argument(
        "--unet_path",
        type=str,
        default=None,
        required=False
    )
    
    parser.add_argument(
        "--max_frame_num",
        type=int,
        default=50,
        required=False
    )

    parser.add_argument(
        "--ref_image_path",
        type=str,
        default=None,
        required=True
    )

    parser.add_argument(
        "--batch_frames",
        type=int,
        default=14,
        required=False
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=4,
        required=False
    )

    parser.add_argument(
        "--sample_stride",
        type=int,
        default=2,
        required=False
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        required=True
    )

    parser.add_argument(
        "--seed",
        type=int,   
        default=23,
    )

    args = parser.parse_args()
    return args



# Main script
if __name__ == "__main__":
    args = parse_args()
    deepspeed_unwrapper(args.checkpoint_dir, args.output_dir)
    args.controlnext_path = os.path.join(args.output_dir, "controlnext", "diffusion_pytorch_model.bin")
    args.unet_path = os.path.join(args.output_dir, "unet", "diffusion_pytorch_model.bin")
    controlnext_inference_runner(args)