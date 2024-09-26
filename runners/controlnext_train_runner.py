import logging
import math
import os
from PIL import Image
from pathlib import Path

import torch
import torch.utils.checkpoint
from torch.utils.data import RandomSampler
import transformers
from accelerate import Accelerator, DistributedType
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo
from packaging import version
from tqdm.auto import tqdm
from safetensors.torch import load_file
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, CLIPTokenizer

from accelerate.logging import get_logger

import diffusers
from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler, UNetSpatioTemporalConditionModel
from diffusers.image_processor import VaeImageProcessor
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import deprecate, is_wandb_available, check_min_version
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import randn_tensor

from dataset_loaders.youtube_vos import make_train_dataset
from models.unet_spatio_temporal_condition_controlnext import UNetSpatioTemporalConditionControlNeXtModel
from models.controlnext_vid_svd import ControlNeXtSVDModel
from pipeline.pipeline_stable_video_diffusion_controlnext import StableVideoDiffusionPipelineControlNeXt
from utils.data_utils import _resize_with_antialiasing, tensor_to_vae_latent, rand_cosine_interpolated, load_images_from_folder, save_combined_frames
from utils.deepspeed_utils import DeepSpeedWrapperModel

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.24.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def controlnext_train_runner(args):
    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir)
    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )

    generator = torch.Generator(
        device=accelerator.device).manual_seed(23123134)

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load scheduler, tokenizer and models.
    noise_scheduler = EulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler")
    feature_extractor = CLIPImageProcessor.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="feature_extractor", revision=args.revision
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="image_encoder", revision=args.revision #, variant="fp16"
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
        
    unet = UNetSpatioTemporalConditionControlNeXtModel.from_pretrained(
        args.pretrained_model_name_or_path if args.pretrain_unet is None else args.pretrain_unet,
        subfolder="unet",
        low_cpu_mem_usage=True,
    )
    
    logger.info("Initializing controlnext weights from unet")
    controlnext = ControlNeXtSVDModel()

    if args.controlnet_model_name_or_path:
        logger.info("Loading existing controlnet weights")
        if args.controlnet_model_name_or_path.endswith(".pth") or args.controlnet_model_name_or_path.endswith(".bin"):
            state_dict = torch.load(args.controlnet_model_name_or_path)
        else:
            state_dict = load_file(args.controlnet_model_name_or_path)
        controlnext.load_state_dict(state_dict, strict=False)

    if args.unet_model_name_or_path:
        logger.info("Loading existing unet weights")
        if args.unet_model_name_or_path.endswith(".pth") or args.controlnet_model_name_or_path.endswith(".bin"):
            state_dict = torch.load(args.unet_model_name_or_path)
        else:
            state_dict = load_file(args.unet_model_name_or_path)
        unet.load_state_dict(state_dict, strict=False)

    # Freeze vae and image_encoder
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    image_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    if args.use_ema:
        ema_controlnext = EMAModel(unet.parameters(
        ), model_cls=UNetSpatioTemporalConditionModel, model_config=unet.config)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly")


    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps *
            args.per_gpu_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        ds_wrapper = DeepSpeedWrapperModel(
            unet=unet,
            controlnext=controlnext
        )
        unet = ds_wrapper.unet
        controlnext = ds_wrapper.controlnext
    controlnext.requires_grad_(True)
    parameters_list = []

    for name, para in controlnext.named_parameters():
        para.requires_grad = True
        parameters_list.append({"params": para, "lr": args.learning_rate } )
    
    """
    For more details, please refer to: https://github.com/dvlab-research/ControlNeXt/issues/14#issuecomment-2290450333
    This is the selective parameters part.
    As presented in our paper, we only select a small subset of parameters, which is fully adapted to the SD1.5 and SDXL backbones. By training fewer than 100 million parameters, we still achieve excellent performance. But this is is not suitable for the SD3 and SVD training. This is because, after SDXL, Stability faced significant legal risks due to the generation of highly realistic human images. After that, they stopped refining their models on human-related data, such as SVD and SD3, to avoid potential risks.
    To achieve optimal performance, it's necessary to first continue training SVD and SD3 on human-related data to develop a robust backbone before fine-tuning. Of course, you can also combine the continual pretraining and finetuning. So you can find that we direct provide the full SVD parameters.
    We have experimented with two approaches: 1.Directly training the model from scratch on human dancing data. 2. Continual training using a pre-trained human generation backbone, followed by fine-tuning a selective small subset of parameters. Interestingly, we observed no significant difference in performance between these two methods.
    """
    if args.finetune_unet:
        for name, para in unet.named_parameters():
            ## For Finetuning of selective parameters
            #if 'to_out' in name or 'to_v' in name:  
            ## For Pretraining 
            if 'to_out' in name or 'to_v' in name:
                para.requires_grad = True
                parameters_list.append({"params": para, "lr": args.learning_rate})


    # Count the number of parameters in parameters_list
    total_params = sum(p.numel() for group in parameters_list for p in group['params'])
    print(f"Total number of trainable parameters: {total_params:,}")
    optimizer = optimizer_cls(
        parameters_list,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # check para
    if accelerator.is_main_process and args.log_trainable_parameters:
        rec_txt1 = open('rec_para.txt', 'w')
        rec_txt2 = open('rec_para_train.txt', 'w')
        for name, para in controlnext.named_parameters():
            if para.requires_grad is False:
                rec_txt1.write(f'{name}\n')
            else:
                rec_txt2.write(f'{name}\n')
        rec_txt1.close()
        rec_txt2.close()
    # DataLoaders creation:
    args.global_batch_size = args.per_gpu_batch_size * accelerator.num_processes

    # Load the tokenizer, feature extractor, and image processor
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")            # Use the one from I2VGen-XL
    feature_extractor = CLIPImageProcessor.from_pretrained(args.pretrained_model_name_or_path, subfolder="feature_extractor")
    image_processor = VaeImageProcessor(vae_scale_factor=2 ** (len(vae.config.block_out_channels) - 1), do_resize=False)

    train_dataset = make_train_dataset(args, tokenizer=tokenizer, feature_extractor=feature_extractor, image_processor=image_processor, annotation_path=args.annotation_path)
    sampler = RandomSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.per_gpu_batch_size,
        num_workers=args.num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        optimizer, lr_scheduler, train_dataloader, ds_wrapper = accelerator.prepare(
            optimizer, lr_scheduler, train_dataloader, ds_wrapper
        )
    else:
        optimizer, lr_scheduler, train_dataloader, unet, controlnext = accelerator.prepare(
            optimizer, lr_scheduler, train_dataloader, unet, controlnext
        )

    if args.use_ema:
        ema_controlnext.to(accelerator.device)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(
        args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("SVDXtend", config=vars(args))

    # Train!
    total_batch_size = args.per_gpu_batch_size * \
        accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_gpu_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    
    def encode_image(pixel_values):
        pixel_values = _resize_with_antialiasing(pixel_values, (224, 224))
        pixel_values = (pixel_values + 1.0) / 2.0

        pixel_values = pixel_values.to(torch.float32)
        # Normalize the image with for CLIP input
        pixel_values = feature_extractor(
            images=pixel_values,
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt",
        ).pixel_values

        pixel_values = pixel_values.to(
            device=accelerator.device, dtype=image_encoder.dtype)
        image_embeddings = image_encoder(pixel_values).image_embeds
        image_embeddings= image_embeddings.unsqueeze(1)
        return image_embeddings


    def _get_add_time_ids(
        fps,
        motion_bucket_id,
        noise_aug_strength,
        dtype,
        batch_size,
        unet=None,
        device=None
    ):
        add_time_ids = [fps, motion_bucket_id, noise_aug_strength]

        
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype, device=device)
        add_time_ids = add_time_ids.repeat(batch_size, 1)
        return add_time_ids

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps),
                        disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        controlnext.train()
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(controlnext, unet):
                # We want to learn the denoising process w.r.t the edited images which
                # are conditioned on the original image (which was edited) and the edit instruction.
                # So, first, convert images to latent space.
                with accelerator.autocast():
                    pixel_values = batch["pixel_values"].to(weight_dtype).to(
                        accelerator.device, non_blocking=True
                    )
                    conditional_pixel_values = batch["reference_image"].to(weight_dtype).to(
                        accelerator.device, non_blocking=True
                    )
                    latents = tensor_to_vae_latent(pixel_values, vae).to(dtype=weight_dtype)

                    # Get the text embedding for conditioning.
                    encoder_hidden_states = encode_image(conditional_pixel_values).to(dtype=weight_dtype)

                    train_noise_aug = 0.02
                    conditional_pixel_values = conditional_pixel_values + train_noise_aug * \
                        randn_tensor(conditional_pixel_values.shape, generator=generator, device=conditional_pixel_values.device, dtype=conditional_pixel_values.dtype)
                    conditional_latents = tensor_to_vae_latent(conditional_pixel_values, vae, scale=False)

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    sigmas = rand_cosine_interpolated(shape=[bsz,], image_d=args.image_d, noise_d_low=args.noise_d_low, noise_d_high=args.noise_d_high, sigma_data=args.sigma_data, min_value=args.min_value, max_value=args.max_value).to(latents.device, dtype=weight_dtype)

                    # sigmas = rand_log_normal(shape=[bsz,], loc=0.7, scale=1.6).to(latents)
                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    sigmas_reshaped = sigmas.clone()
                    while len(sigmas_reshaped.shape) < len(latents.shape):
                        sigmas_reshaped = sigmas_reshaped.unsqueeze(-1)


                    noisy_latents  = latents + noise * sigmas_reshaped
                    
                    timesteps = torch.Tensor([0.25 * sigma.log() for sigma in sigmas]).to(latents.device, dtype=weight_dtype)

                    
                    inp_noisy_latents = noisy_latents  / ((sigmas_reshaped**2 + 1) ** 0.5)
                    
                    added_time_ids = _get_add_time_ids(
                        fps=6,
                        motion_bucket_id=127.0,
                        noise_aug_strength=train_noise_aug, # noise_aug_strength == 0.0
                        dtype=encoder_hidden_states.dtype,
                        batch_size=bsz,
                        unet=unet,
                        device=latents.device
                    )

                    added_time_ids = added_time_ids.to(latents.device)

                    # Conditioning dropout to support classifier-free guidance during inference. For more details
                    # check out the section 3.2.1 of the original paper https://arxiv.org/abs/2211.09800.
                    if args.conditioning_dropout_prob is not None:
                        random_p = torch.rand(
                            bsz, device=latents.device, generator=generator)
                        # Sample masks for the edit prompts.
                        prompt_mask = random_p < 2 * args.conditioning_dropout_prob
                        prompt_mask = prompt_mask.reshape(bsz, 1, 1)
                        # Final text conditioning.
                        null_conditioning = torch.zeros_like(encoder_hidden_states)
                        encoder_hidden_states = torch.where(
                            prompt_mask, null_conditioning, encoder_hidden_states)

                        # Sample masks for the original images.
                        image_mask_dtype = conditional_latents.dtype
                        image_mask = 1 - (
                            (random_p >= args.conditioning_dropout_prob).to(
                                image_mask_dtype)
                            * (random_p < 3 * args.conditioning_dropout_prob).to(image_mask_dtype)
                        )
                        image_mask = image_mask.reshape(bsz, 1, 1, 1)
                        # Final image conditioning.
                        conditional_latents = image_mask * conditional_latents

                    # Concatenate the `conditional_latents` with the `noisy_latents`.
                    conditional_latents = conditional_latents.unsqueeze(
                        1).repeat(1, noisy_latents.shape[1], 1, 1, 1)
                    controlnext_image = batch["guide_values"].to(
                        dtype=weight_dtype, device=accelerator.device, non_blocking=True
                    )
                    controlnext_output = controlnext(controlnext_image, timesteps)
                    

                    inp_noisy_latents = torch.cat(
                        [inp_noisy_latents, conditional_latents], dim=2)
                    target = latents
                
                    # Predict the noise residual
                    model_pred = unet(
                        inp_noisy_latents, timesteps, encoder_hidden_states,
                        added_time_ids=added_time_ids,
                        conditional_controls=controlnext_output,
                        ).sample
                    

                    sigmas = sigmas_reshaped
                    # Denoise the latents
                    c_out = -sigmas / ((sigmas**2 + 1)**0.5)
                    c_skip = 1 / (sigmas**2 + 1)
                    denoised_latents = model_pred * c_out + c_skip * noisy_latents
                    weighing = (1 + sigmas ** 2) * (sigmas**-2.0)

                    # MSE losss
                    loss = torch.mean(
                        (weighing.float() * (denoised_latents.float() -
                        target.float()) ** 2).reshape(target.shape[0], -1),
                        dim=1,
                    )
                    loss = loss.mean()

                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss = accelerator.gather(
                        loss.repeat(args.per_gpu_batch_size)).mean()
                    train_loss += avg_loss.item() / args.gradient_accumulation_steps

                    # Backpropagate
                    accelerator.backward(loss)
                    # if accelerator.sync_gradients:
                    #     accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_controlnext.step(controlnext.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                # save checkpoints!
                if global_step % args.checkpointing_steps == 0 and (accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED):
                    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                    if args.checkpoints_total_limit is not None and accelerator.is_main_process:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [
                            d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(
                            checkpoints, key=lambda x: int(x.split("-")[1]))

                        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = len(
                                checkpoints) - args.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]

                            logger.info(
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                            )
                            logger.info(
                                f"removing checkpoints: {', '.join(removing_checkpoints)}")

                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(
                                    args.output_dir, removing_checkpoint)

                    save_path = os.path.join(
                        args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

                if accelerator.is_main_process:
                    # sample images!
                    if global_step % args.validation_steps == 0:
                        logger.info(
                            f"Running validation... \n Generating {args.num_validation_images} videos."
                        )
                        # create pipeline
                        if args.use_ema:
                            # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                            ema_controlnext.store(controlnext.parameters())
                            ema_controlnext.copy_to(controlnext.parameters())
                        # The models need unwrapping because for compatibility in distributed training mode.
                        pipeline = StableVideoDiffusionPipelineControlNeXt.from_pretrained(
                            args.pretrained_model_name_or_path,
                            unet=accelerator.unwrap_model(unet),
                            controlnext=accelerator.unwrap_model(
                                controlnext),
                            image_encoder=accelerator.unwrap_model(
                                image_encoder),
                            vae=accelerator.unwrap_model(vae),
                            revision=args.revision,
                            torch_dtype=weight_dtype,
                        )
                        pipeline = pipeline.to(accelerator.device)
                        pipeline.set_progress_bar_config(disable=True)
                        
                        validation_images = load_images_from_folder(args.validation_image_folder)
                        if args.validation_image is None:
                            validation_image = validation_images[0]
                        else:
                            validation_image = Image.open(args.validation_image).convert('RGB')

                        validation_control_images = load_images_from_folder(args.validation_control_folder)
                        
                        # run inference
                        val_save_dir = os.path.join(
                            args.output_dir, "validation_images")

                        if not os.path.exists(val_save_dir):
                            os.makedirs(val_save_dir)

                        with accelerator.autocast():
                            for val_img_idx in range(args.num_validation_images):
                                # num_frames = args.num_frames
                                num_frames = len(validation_control_images)
                                video_frames = pipeline(
                                    validation_image, 
                                    validation_control_images,
                                    height=args.height,
                                    width=args.width,
                                    num_frames=num_frames,
                                    frames_per_batch=14, #args.sample_n_frames,
                                    decode_chunk_size=4,
                                    motion_bucket_id=127.,
                                    fps=7,
                                    controlnext_cond_scale=1.0,
                                    min_guidance_scale=3, 
                                    max_guidance_scale=3, 
                                    noise_aug_strength=0.02,
                                    num_inference_steps=25,
                                    overlap=4,
                                ).frames
                                save_combined_frames(video_frames, validation_images, validation_control_images, val_save_dir, step=global_step)
        

                        if args.use_ema:
                            # Switch back to the original UNet parameters.
                            ema_controlnext.restore(controlnext.parameters())

                        del pipeline
                        torch.cuda.empty_cache()

            logs = {"step_loss": loss.detach().item(
            ), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

            
    # save checkpoints!
    if accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED:
        save_path = os.path.join(
            args.output_dir, f"checkpoint-last")
        accelerator.save_state(save_path)
        logger.info(f"Saved state to {save_path}")