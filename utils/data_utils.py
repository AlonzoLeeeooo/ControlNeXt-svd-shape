import os
import re
import cv2
import PIL
import math
import datetime
import numpy as np
from PIL import Image 
from einops import rearrange
from urllib.parse import urlparse
from typing import List, Tuple, Union
from decord import VideoReader
from moviepy.editor import ImageSequenceClip
from safetensors.torch import load_file

import torch 
from torchvision import transforms

from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import load_image


def center_crop_and_resize(img, output_size=(512, 512)):
    # Load the image
    #img = Image.open(image_path)
    
    # Calculate the aspect ratio of the output image
    aspect_ratio = output_size[0] / output_size[1]
    
    # Get the current size of the image
    original_width, original_height = img.size
    
    # Calculate the aspect ratio of the original image
    original_aspect_ratio = original_width / original_height
    
    # Determine the dimensions to which the image needs to be resized before cropping
    if original_aspect_ratio > aspect_ratio:
        # Image is wider than the desired aspect ratio; resize based on height
        new_height = output_size[1]
        new_width = int(new_height * original_aspect_ratio)
    else:
        # Image is taller than the desired aspect ratio; resize based on width
        new_width = output_size[0]
        new_height = int(new_width / original_aspect_ratio)
    
    # Resize the image
    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Calculate the cropping box
    left = (new_width - output_size[0]) / 2
    top = (new_height - output_size[1]) / 2
    right = (new_width + output_size[0]) / 2
    bottom = (new_height + output_size[1]) / 2
    
    # Crop the center
    img_cropped = img_resized.crop((left, top, right, bottom))
    
    return img_cropped


image_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


def image_to_tensor(img):
    
    with torch.no_grad():
        if img.mode != "RGB":
            img = img.convert("RGB")

        image = image_transforms(img)#.to(accelerator.device)

        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        if image.shape[0] > 3:
            image = image[:3, :, :]

    return image


# NOTE: Newly defined functions
def binarize_tensor(tensor):
    with torch.no_grad():
        tensor[tensor > 0.5] = 1
        tensor[tensor < 0.5] = 0
        
        return tensor
    


# The following utilities are taken and adapted from
# https://github.com/ali-vilab/i2vgen-xl/blob/main/utils/transforms.py.
def _convert_pt_to_pil(image: Union[torch.Tensor, List[torch.Tensor]]):
    if isinstance(image, list) and isinstance(image[0], torch.Tensor):
        image = torch.cat(image, 0)

    if isinstance(image, torch.Tensor):
        if image.ndim == 3:
            image = image.unsqueeze(0)

        image_numpy = VaeImageProcessor.pt_to_numpy(image)
        image_pil = VaeImageProcessor.numpy_to_pil(image_numpy)
        image = image_pil

    return image


def _resize_bilinear(
    image: Union[torch.Tensor, List[torch.Tensor], Image.Image, List[Image.Image]], resolution: Tuple[int, int]
):
    # First convert the images to PIL in case they are float tensors (only relevant for tests now).
    image = _convert_pt_to_pil(image)

    if isinstance(image, list):
        image = [u.resize(resolution, Image.BILINEAR) for u in image]
    else:
        image = image.resize(resolution, Image.BILINEAR)
    return image


#i should make a utility function file
def validate_and_convert_image(image, target_size=(256, 256)):
    if image is None:
        print("Encountered a None image")
        return None

    if isinstance(image, torch.Tensor):
        # Convert PyTorch tensor to PIL Image
        if image.ndim == 3 and image.shape[0] in [1, 3]:  # Check for CxHxW format
            if image.shape[0] == 1:  # Convert single-channel grayscale to RGB
                image = image.repeat(3, 1, 1)
            image = image.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            image = Image.fromarray(image)
        else:
            print(f"Invalid image tensor shape: {image.shape}")
            return None
    elif isinstance(image, Image.Image):
        # Resize PIL Image
        image = image.resize(target_size)
    else:
        print("Image is not a PIL Image or a PyTorch tensor")
        return None
    
    return image

def create_image_grid(images, rows, cols, target_size=(256, 256)):
    valid_images = [validate_and_convert_image(img, target_size) for img in images]
    valid_images = [img for img in valid_images if img is not None]

    if not valid_images:
        print("No valid images to create a grid")
        return None

    w, h = target_size
    grid = Image.new('RGB', size=(cols * w, rows * h))

    for i, image in enumerate(valid_images):
        grid.paste(image, box=((i % cols) * w, (i // cols) * h))

    return grid

def save_combined_frames(batch_output, validation_images, validation_control_images, output_folder, step=None):
    # Flatten batch_output, which is a list of lists of PIL Images
    flattened_batch_output = [img for sublist in batch_output for img in sublist]

    # Combine frames into a list without converting (since they are already PIL Images)
    combined_frames = validation_images + validation_control_images + flattened_batch_output

    # Calculate rows and columns for the grid
    num_images = len(combined_frames)
    cols = 3  # adjust number of columns as needed
    rows = (num_images + cols - 1) // cols

    filename = f"step_{step}.png"
    # Create and save the grid image
    grid = create_image_grid(combined_frames, rows, cols)
    output_folder = os.path.join(output_folder, "validation_images")
    os.makedirs(output_folder, exist_ok=True)
    
    # Now define the full path for the file
    filename = f"step_{step}.png"
    output_loc = os.path.join(output_folder, filename)
    
    if grid is not None:
        grid.save(output_loc)
    else:
        print("Failed to create image grid")



def load_images_from_folder(folder):
    images = []
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}  # Add or remove extensions as needed

    # Function to extract frame number from the filename
    def frame_number(filename):
        # First, try the pattern 'frame_x_7fps'
        new_pattern_match = re.search(r'frame_(\d+)_7fps', filename)
        if new_pattern_match:
            return int(new_pattern_match.group(1))
        # If the new pattern is not found, use the original digit extraction method
        matches = re.findall(r'\d+', filename)
        if matches:
            if matches[-1] == '0000' and len(matches) > 1:
                return int(matches[-2])  # Return the second-to-last sequence if the last is '0000'
            return int(matches[-1])  # Otherwise, return the last sequence
        return float('inf')  # Return 'inf'

    # Sorting files based on frame number
    sorted_files = sorted(os.listdir(folder), key=frame_number)

    # Load images in sorted order
    for filename in sorted_files:
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_extensions:
            img = Image.open(os.path.join(folder, filename)).convert('RGB')
            images.append(img)

    return images



# copy from https://github.com/crowsonkb/k-diffusion.git
def stratified_uniform(shape, group=0, groups=1, dtype=None, device=None):
    """Draws stratified samples from a uniform distribution."""
    if groups <= 0:
        raise ValueError(f"groups must be positive, got {groups}")
    if group < 0 or group >= groups:
        raise ValueError(f"group must be in [0, {groups})")
    n = shape[-1] * groups
    offsets = torch.arange(group, n, groups, dtype=dtype, device=device)
    u = torch.rand(shape, dtype=dtype, device=device)
    return (offsets + u) / n


def rand_cosine_interpolated(shape, image_d, noise_d_low, noise_d_high, sigma_data=1., min_value=1e-3, max_value=1e3, device='cpu', dtype=torch.float32):
    """Draws samples from an interpolated cosine timestep distribution (from simple diffusion)."""

    def logsnr_schedule_cosine(t, logsnr_min, logsnr_max):
        t_min = math.atan(math.exp(-0.5 * logsnr_max))
        t_max = math.atan(math.exp(-0.5 * logsnr_min))
        return -2 * torch.log(torch.tan(t_min + t * (t_max - t_min)))

    def logsnr_schedule_cosine_shifted(t, image_d, noise_d, logsnr_min, logsnr_max):
        shift = 2 * math.log(noise_d / image_d)
        return logsnr_schedule_cosine(t, logsnr_min - shift, logsnr_max - shift) + shift

    def logsnr_schedule_cosine_interpolated(t, image_d, noise_d_low, noise_d_high, logsnr_min, logsnr_max):
        logsnr_low = logsnr_schedule_cosine_shifted(
            t, image_d, noise_d_low, logsnr_min, logsnr_max)
        logsnr_high = logsnr_schedule_cosine_shifted(
            t, image_d, noise_d_high, logsnr_min, logsnr_max)
        return torch.lerp(logsnr_low, logsnr_high, t)

    logsnr_min = -2 * math.log(min_value / sigma_data)
    logsnr_max = -2 * math.log(max_value / sigma_data)
    u = stratified_uniform(
        shape, group=0, groups=1, dtype=dtype, device=device
    )
    logsnr = logsnr_schedule_cosine_interpolated(
        u, image_d, noise_d_low, noise_d_high, logsnr_min, logsnr_max)
    return torch.exp(-logsnr / 2) * sigma_data

def rand_log_normal(shape, loc=0., scale=1., device='cpu', dtype=torch.float32):
    """Draws samples from an lognormal distribution."""
    u = torch.rand(shape, dtype=dtype, device=device) * (1 - 2e-7) + 1e-7
    return torch.distributions.Normal(loc, scale).icdf(u).exp()

min_value = 0.002
max_value = 700
image_d = 64
noise_d_low = 32
noise_d_high = 64
sigma_data = 0.5


def _resize_with_antialiasing(input, size, interpolation="bicubic", align_corners=True):
    h, w = input.shape[-2:]
    factors = (h / size[0], w / size[1])

    # First, we have to determine sigma
    # Taken from skimage: https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/transform/_warps.py#L171
    sigmas = (
        max((factors[0] - 1.0) / 2.0, 0.001),
        max((factors[1] - 1.0) / 2.0, 0.001),
    )

    # Now kernel size. Good results are for 3 sigma, but that is kind of slow. Pillow uses 1 sigma
    # https://github.com/python-pillow/Pillow/blob/master/src/libImaging/Resample.c#L206
    # But they do it in the 2 passes, which gives better results. Let's try 2 sigmas for now
    ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))

    # Make sure it is odd
    if (ks[0] % 2) == 0:
        ks = ks[0] + 1, ks[1]

    if (ks[1] % 2) == 0:
        ks = ks[0], ks[1] + 1

    input = _gaussian_blur2d(input, ks, sigmas)

    output = torch.nn.functional.interpolate(
        input, size=size, mode=interpolation, align_corners=align_corners)
    return output


def _compute_padding(kernel_size):
    """Compute padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]

    # for even kernels we need to do asymmetric padding :(
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]

        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front

        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding


def _filter2d(input, kernel):
    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...].to(
        device=input.device, dtype=input.dtype)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    height, width = tmp_kernel.shape[-2:]

    padding_shape: list[int] = _compute_padding([height, width])
    input = torch.nn.functional.pad(input, padding_shape, mode="reflect")

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    # convolve the tensor with the kernel.
    output = torch.nn.functional.conv2d(
        input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

    out = output.view(b, c, h, w)
    return out


def _gaussian(window_size: int, sigma):
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]])

    batch_size = sigma.shape[0]

    x = (torch.arange(window_size, device=sigma.device,
         dtype=sigma.dtype) - window_size // 2).expand(batch_size, -1)

    if window_size % 2 == 0:
        x = x + 0.5

    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))

    return gauss / gauss.sum(-1, keepdim=True)


def _gaussian_blur2d(input, kernel_size, sigma):
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], dtype=input.dtype)
    else:
        sigma = sigma.to(dtype=input.dtype)

    ky, kx = int(kernel_size[0]), int(kernel_size[1])
    bs = sigma.shape[0]
    kernel_x = _gaussian(kx, sigma[:, 1].view(bs, 1))
    kernel_y = _gaussian(ky, sigma[:, 0].view(bs, 1))
    out_x = _filter2d(input, kernel_x[..., None, :])
    out = _filter2d(out_x, kernel_y[..., None])

    return out


def export_to_video(video_frames, output_video_path, fps):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w, _ = video_frames[0].shape
    video_writer = cv2.VideoWriter(
        output_video_path, fourcc, fps=fps, frameSize=(w, h))
    for i in range(len(video_frames)):
        img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
        video_writer.write(img)


def export_to_gif(frames, output_gif_path, fps):
    """
    Export a list of frames to a GIF.

    Args:
    - frames (list): List of frames (as numpy arrays or PIL Image objects).
    - output_gif_path (str): Path to save the output GIF.
    - duration_ms (int): Duration of each frame in milliseconds.

    """
    # Convert numpy arrays to PIL Images if needed
    pil_frames = [Image.fromarray(frame) if isinstance(
        frame, np.ndarray) else frame for frame in frames]

    pil_frames[0].save(output_gif_path.replace('.mp4', '.gif'),
                       format='GIF',
                       append_images=pil_frames[1:],
                       save_all=True,
                       duration=500,
                       loop=0)


def tensor_to_vae_latent(t, vae, scale=True):
    t = t.to(vae.dtype)
    if len(t.shape) == 5:
        video_length = t.shape[1]

        t = rearrange(t, "b f c h w -> (b f) c h w")
        latents = vae.encode(t).latent_dist.sample()
        latents = rearrange(latents, "(b f) c h w -> b f c h w", f=video_length)
    elif len(t.shape) == 4:
        latents = vae.encode(t).latent_dist.sample()
    if scale:
        latents = latents * vae.config.scaling_factor
    return latents

def download_image(url):
    original_image = (
        lambda image_url_or_path: load_image(image_url_or_path)
        if urlparse(image_url_or_path).scheme
        else PIL.Image.open(image_url_or_path).convert("RGB")
    )(url)
    return original_image


def write_mp4(video_path, samples, fps=14, audio_bitrate="192k"):
    clip = ImageSequenceClip(samples, fps=fps)
    clip.write_videofile(video_path, audio_codec="aac", audio_bitrate=audio_bitrate, 
                         ffmpeg_params=["-crf", "18", "-preset", "slow"])

def save_vid_side_by_side(batch_output,
                          validation_control_images,
                          output_folder,
                          fps,
                          filename,
                          seed):
    # Helper function to convert tensors to PIL images and save as GIF
    flattened_batch_output = [img for sublist in batch_output for img in sublist]
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, filename), exist_ok=True)
    video_path = os.path.join(output_folder, filename, f"seed_{seed}_concat.mp4")
    output_path = os.path.join(output_folder, filename, f"seed_{seed}_output.mp4")
    final_images = []
    outputs = []
    # Helper function to concatenate images horizontally
    def get_concat_h(im1, im2):
        dst = Image.new('RGB', (im1.width + im2.width, max(im1.height, im2.height)))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst
    for image_list in zip(validation_control_images, flattened_batch_output):
        predict_img = image_list[1].resize(image_list[0].size)
        result = get_concat_h(image_list[0], predict_img)
        final_images.append(np.array(result))
        outputs.append(np.array(predict_img))
    write_mp4(video_path, final_images, fps=fps)

    write_mp4(output_path, outputs, fps=fps)


def load_images_from_folder_to_pil(folder):
    images = []
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}  # Add or remove extensions as needed

    # Function to extract frame number from the filename
    def frame_number(filename):
        # First, try the pattern 'frame_x_7fps'
        new_pattern_match = re.search(r'frame_(\d+)_7fps', filename)
        if new_pattern_match:
            return int(new_pattern_match.group(1))
        # If the new pattern is not found, use the original digit extraction method
        matches = re.findall(r'\d+', filename)
        if matches:
            if matches[-1] == '0000' and len(matches) > 1:
                return int(matches[-2])  # Return the second-to-last sequence if the last is '0000'
            return int(matches[-1])  # Otherwise, return the last sequence
        return float('inf')  # Return 'inf'

    # Sorting files based on frame number
    sorted_files = sorted(os.listdir(folder), key=frame_number)
    # Load images in sorted order
    for filename in sorted_files:
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_extensions:
            img = Image.open(os.path.join(folder, filename)).convert('RGB')
            images.append(img)

    return images


def load_images_from_video_to_pil(video_path):
    images = []

    vr = VideoReader(video_path)
    length = len(vr)

    for idx in range(length):
        frame = vr[idx].asnumpy()
        images.append(Image.fromarray(frame))
    return images


def load_tensor(tensor_path):
    if os.path.splitext(tensor_path)[1] == '.bin':
        return torch.load(tensor_path)
    elif os.path.splitext(tensor_path)[1] == ".safetensors":
        return load_file(tensor_path)
    else:
        print("without supported tensors")
        os._exit()