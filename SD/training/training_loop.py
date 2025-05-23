# Copyright (c) 2024, Mingyuan Zhou. All rights reserved.
#
# This work is licensed under APACHE LICENSE, VERSION 2.0
# You should have received a copy of the license along with this
# work. If not, see https://www.apache.org/licenses/LICENSE-2.0.txt

"""Distill Stable Diffusion models using the SiD-LSG techniques described in the
paper "Long and Short Guidance in Score identity Distillation for One-Step Text-to-Image Generation"."""

"""Main training loop."""
import re
import os
import time
import copy
import json
import pickle
import psutil
import PIL.Image
import numpy as np
import torch
import dnnlib
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc
import torch.nn as nn
from functools import partial
import gc

# Needed for v-prediction based diffusion model
from diffusers.training_utils import compute_snr

# Functions needed to integrate Stable Diffusion into SiD
from training.sd_util import load_sd15, sid_sd_sampler, sid_sd_denoise


# ----------------------------------------------------------------------------
def setup_snapshot_image_grid(training_set, random_seed=0):
    gw = np.clip(4096 // training_set.resolution, 8, 32)
    gh = np.clip(2048 // training_set.resolution, 4, 32)
    all_indices = list(range(len(training_set)))

    if random_seed is not None:
        np.random.RandomState(random_seed).shuffle(all_indices)

    _gw = gw // 2
    grid_indices = [all_indices[i % len(all_indices)] for i in range(_gw * gh)]

    contexts = []
    for i in grid_indices:
        contexts.extend([training_set[i][0], training_set[i][0]])

    return (gw, gh), None, contexts

from itertools import islice


def split_list(lst, split_sizes):
    """
    Splits a list into chunks based on split_sizes.

    Parameters:
    - lst (list): The list to be split.
    - split_sizes (list or int): Sizes of the chunks to split the list into.
                                 If it's an integer, the list will be divided into chunks of this size.
                                 If it's a list of integers, the list will be divided into chunks of varying sizes specified by the list.

    Returns:
    - list of lists: The split list.
    """
    if isinstance(split_sizes, int):
        # If split_sizes is an integer, create a list of sizes to split the list evenly, except the last chunk which may be smaller.
        split_sizes = [split_sizes] * (len(lst) // split_sizes) + (
            [len(lst) % split_sizes] if len(lst) % split_sizes != 0 else [])
    it = iter(lst)
    return [list(islice(it, size)) for size in split_sizes]


from PIL import Image


def save_pil_images_in_grid(image_files, grid_size, output_path):
    gw, gh = grid_size
    # Assuming all images are the same size, open the first image to get its size
    image_width, image_height = image_files[0].size

    # Calculate the total grid size
    grid_width = gw * image_width
    grid_height = gh * image_height

    # Create a new blank image for the grid
    grid_image = Image.new('RGB', (grid_width, grid_height))

    # Iterate over the images and paste them into the grid
    for index, image in enumerate(image_files):
        # Calculate the position based on the index
        x = (index % gw) * image_width
        y = (index // gw) * image_height
        grid_image.paste(image, (x, y))

    # Save the final grid image
    grid_image.save(output_path)


# ----------------------------------------------------------------------------
# Helper methods


def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape(gh, gw, C, H, W)
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape(gh * H, gw * W, C)

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)


def save_data(data, fname):
    with open(fname, 'wb') as f:
        pickle.dump(data, f)


def save_pt(pt, fname):
    torch.save(pt, fname)


def append_line(jsonl_line, fname):
    with open(fname, 'at') as f:
        f.write(jsonl_line + '\n')


# ----------------------------------------------------------------------------

def training_loop(
        run_dir='.',  # Output directory.
        dataset_kwargs={},  # Options for training set.
        data_loader_kwargs={},  # Options for torch.utils.data.DataLoader.
        network_kwargs={},  # Options for model and preconditioning.
        loss_kwargs={},  # Options for loss function.
        fake_score_optimizer_kwargs={},  # Options for fake score network optimizer.
        g_optimizer_kwargs={},  # Options for generator optimizer.
        augment_kwargs=None,  # Options for augmentation pipeline, None = disable.
        seed=0,  # Global random seed.
        batch_size=512,  # Total batch size for one training iteration.
        batch_gpu=None,  # Limit batch size per GPU, None = no limit.
        total_kimg=200000,  # Training duration, measured in thousands of training images.
        ema_halflife_kimg=500,  # Half-life of the exponential moving average (EMA) of model weights.
        ema_rampup_ratio=0.05,  # EMA ramp-up coefficient, None = no rampup.
        loss_scaling=1,  # Loss scaling factor, could be adjusted for reducing FP16 under/overflows.
        loss_scaling_G=1,  # Loss scaling factor of G, could be adjusted for reducing FP16 under/overflows.
        kimg_per_tick=50,  # Interval of progress prints.
        snapshot_ticks=50,  # How often to save network snapshots, None = disable.
        state_dump_ticks=500,  # How often to dump training state, None = disable.
        resume_pkl=None,  # Start from the given network snapshot for initialization, None = random initialization.
        resume_training=None,  # Resume training from the given network snapshot.
        resume_kimg=0,  # Start from the given training progress.
        alpha=1,  # loss = L2-alpha*L1
        tmax=980,  # We add noise at steps 0 to tmax, tmax <= 1000
        tmin=20,  # We add noise at steps 0 to tmax, tmax <= 1000
        cudnn_benchmark=True,  # Enable torch.backends.cudnn.benchmark?
        device=torch.device('cuda'),
        init_timestep=None,
        pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
        dataset_prompt_text_kwargs={},
        forget_dataset_prompt_text_kwargs={},
        cfg_train_fake=1,  # kappa1
        cfg_eval_fake=1,  # kappa2 = kappa3
        cfg_eval_real=1,  # kappa4
        num_steps=1,
        enable_xformers=True,
        gradient_checkpointing=False,
        resolution=512,
        sg_remain_coef=1.0,
        sg_forget_coef=0.01,
        g_remain_coef=1.0,
        g_forget_coef=0.01,
        from_distill_ema=None,
        sid_w_neg=False,
        use_neg=(False, False, True),
        sg_w_override=False,
):
    # Load dataset.
    dist.print0('Loading dataset...')
    _forget_dataset_prompt_text_kwargs = copy.deepcopy(forget_dataset_prompt_text_kwargs)
    _forget_dataset_prompt_text_kwargs["max_size"] = 8
    _forget_dataset_prompt_text_kwargs["random_seed"] = None
    _forget_dataset_prompt_text_kwargs["path_to_neg"] = None
    forget_dataset_prompt_text_kwargs["path_to_val"] = None
    dataset_obj = dnnlib.util.construct_class_by_name(
        **_forget_dataset_prompt_text_kwargs)  # subclass of training.dataset.Dataset

    dtype = torch.float16 if network_kwargs.use_fp16 else torch.float32

    if cfg_train_fake != 1 or cfg_eval_fake != 1:
        use_context_dropout_train_fake = True
    else:
        use_context_dropout_train_fake = False
    use_context_dropout_train_G = False

    # Score Forgetting Distillation of Stable Diffusion
    # Use barrier if it needs to download the weights from internet and save to cache
    if dist.get_rank() != 0:
        torch.distributed.barrier()
    if dtype == torch.float16:
        # if the fp16 checkpoint variant is not available, you can load the fp32 version and then convert them into fp16
        unet, vae, noise_scheduler, text_encoder, tokenizer, unet_cpu_copy = load_sd15(
            pretrained_model_name_or_path=pretrained_model_name_or_path, pretrained_vae_model_name_or_path=None,
            device=device, weight_dtype=dtype, variant="fp16", enable_xformers=enable_xformers,
            lora_config=None)
    else:
        unet, vae, noise_scheduler, text_encoder, tokenizer, unet_cpu_copy = load_sd15(
            pretrained_model_name_or_path=pretrained_model_name_or_path, pretrained_vae_model_name_or_path=None,
            device=device, weight_dtype=dtype, enable_xformers=enable_xformers, lora_config=None)

    if dist.get_rank() == 0:
        torch.distributed.barrier()
    dist.print0('Loading network completed')
    dist.print0('Noise scheduler:')
    dist.print0(noise_scheduler)

    # Initialize.
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Select batch size per GPU. Used for gradient accumulation
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    # Parameters for latent diffusion
    latent_img_channels = 4
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    latent_resolution = resolution // vae_scale_factor

    # Prepare the random noise used for example image generation during training
    if dist.get_rank() == 0:
        original_seed = torch.initial_seed()

        # Set a temporary random seed
        temporary_seed = 2024
        torch.manual_seed(temporary_seed)
        grid_size, _, contexts = setup_snapshot_image_grid(training_set=dataset_obj, random_seed=None)
        print("Snapshot contexts:", contexts)
        # contexts = [""] * len(contexts)
        grid_z = torch.randn([len(contexts), latent_img_channels, latent_resolution, latent_resolution],
                             device=device, dtype=dtype)
        grid_z = grid_z.split(batch_gpu)
        grid_c = split_list(contexts, batch_gpu)
        # Revert back to the original random seed
        torch.manual_seed(original_seed)

    dataset_prompt_text_obj = dnnlib.util.construct_class_by_name(
        **dataset_prompt_text_kwargs)  # subclass of training.dataset.Dataset
    dataset_prompt_text_sampler = misc.InfiniteSampler(dataset=dataset_prompt_text_obj, rank=dist.get_rank(),
                                                       num_replicas=dist.get_world_size(), seed=seed)
    dataset_prompt_text_iterator = iter(
        torch.utils.data.DataLoader(dataset=dataset_prompt_text_obj, sampler=dataset_prompt_text_sampler,
                                    batch_size=batch_gpu, **data_loader_kwargs))
    forget_dataset_prompt_text_obj = dnnlib.util.construct_class_by_name(
        **forget_dataset_prompt_text_kwargs)  # subclass of training.dataset.Dataset
    forget_dataset_prompt_text_sampler = misc.InfiniteSampler(dataset=forget_dataset_prompt_text_obj,
                                                              rank=dist.get_rank(),
                                                              num_replicas=dist.get_world_size(), seed=seed)
    forget_dataset_prompt_text_iterator = iter(torch.utils.data.DataLoader(dataset=forget_dataset_prompt_text_obj,
                                                                           sampler=forget_dataset_prompt_text_sampler,
                                                                           batch_size=batch_gpu,
                                                                           **data_loader_kwargs))

    dist.print0("Example text prompts used for distillation:")
    for _i in range(16):
        dist.print0(_i)
        _, contexts = next(dataset_prompt_text_iterator)
        dist.print0(contexts)

    # Initilize true score net, fake score net, and generator
    true_score = unet
    true_score.eval().requires_grad_(False).to(device)
    fake_score = copy.deepcopy(true_score).train().requires_grad_(True).to(device)
    G = copy.deepcopy(true_score).train().requires_grad_(True).to(device)

    # Setup optimizer.
    dist.print0('Setting up optimizer...')
    fake_score_optimizer = dnnlib.util.construct_class_by_name(params=fake_score.parameters(),
                                                               **fake_score_optimizer_kwargs)  # subclass of torch.optim.Optimizer
    g_optimizer = dnnlib.util.construct_class_by_name(params=G.parameters(),
                                                      **g_optimizer_kwargs)  # subclass of torch.optim.Optimizer

    if from_distill_ema is not None:
        dist.print0('checkpoint path:', from_distill_ema)
        with open(from_distill_ema, "rb") as f:
            data = pickle.load(f)
        misc.copy_params_and_buffers(src_module=data["ema"], dst_module=G, require_all=True)

    # Resume training from previous snapshot.
    if resume_training is not None:
        dist.print0('checkpoint path:', resume_training)
        data = torch.load(resume_training, map_location=torch.device('cpu'))
        misc.copy_params_and_buffers(src_module=data['fake_score'], dst_module=fake_score, require_all=True)
        misc.copy_params_and_buffers(src_module=data['G'], dst_module=G, require_all=True)
        if ema_halflife_kimg > 0:
            G_ema = copy.deepcopy(G).eval().requires_grad_(False)
            misc.copy_params_and_buffers(src_module=data['G_ema'], dst_module=G_ema, require_all=True)
            G_ema.eval().requires_grad_(False)
        else:
            G_ema = G
        fake_score_optimizer.load_state_dict(data['fake_score_optimizer_state'])
        g_optimizer.load_state_dict(data['g_optimizer_state'])
        del data  # conserve memory
        dist.print0('Loading checkpoint completed')

        torch.distributed.barrier()

        # Setup GPU parallel computing.
        dist.print0('Setting up GPU parallel computing')
        fake_score_ddp = torch.nn.parallel.DistributedDataParallel(fake_score, device_ids=[device],
                                                                   broadcast_buffers=False,
                                                                   find_unused_parameters=False)
        G_ddp = torch.nn.parallel.DistributedDataParallel(G, device_ids=[device], broadcast_buffers=False,
                                                          find_unused_parameters=False)

    else:
        # Setup GPU parallel computing.
        dist.print0('Setting up GPU parallel computing')
        fake_score_ddp = torch.nn.parallel.DistributedDataParallel(fake_score, device_ids=[device],
                                                                   broadcast_buffers=False,
                                                                   find_unused_parameters=False)
        G_ddp = torch.nn.parallel.DistributedDataParallel(G, device_ids=[device], broadcast_buffers=False,
                                                          find_unused_parameters=False)
        if ema_halflife_kimg > 0:
            G_ema = copy.deepcopy(G).eval().requires_grad_(False)
        else:
            G_ema = G

    fake_score_ddp.eval().requires_grad_(False)
    G_ddp.eval().requires_grad_(False)

    # Train.
    dist.print0(f'Training for {total_kimg} kimg...')
    dist.print0()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None

    if resume_training is None:
        if dist.get_rank() == 0:
            print('Text prompts for example images:')
            for c in grid_c:
                dist.print0(c)

            print('Exporting sample fake images at initialization...')
            images = [sid_sd_sampler(unet=G_ema, latents=z, contexts=c,
                                     init_timesteps=init_timestep * torch.ones((len(c),), device=device,
                                                                               dtype=torch.long),
                                     noise_scheduler=noise_scheduler,
                                     text_encoder=text_encoder, tokenizer=tokenizer,
                                     resolution=resolution, dtype=dtype, return_images=True, vae=vae,
                                     num_steps=num_steps, train_sampler=False, num_steps_eval=1) for z, c in
                      zip(grid_z, grid_c)]
            images = torch.cat(images).cpu().numpy()
            save_image_grid(img=images, fname=os.path.join(run_dir, 'fakes_init.png'), drange=[-1, 1],
                            grid_size=grid_size)
            del images

    torch.distributed.barrier()

    dist.print0('Start Running')
    while True:
        torch.cuda.empty_cache()
        gc.collect()
        G_ddp.eval().requires_grad_(False)
        # ----------------------------------------------------------------------------------------------
        # Update Fake Score Network
        fake_score_ddp.train().requires_grad_(True)
        fake_score_optimizer.zero_grad(set_to_none=True)
        sg_remain_loss_print = sg_forget_loss_print = 0
        for round_idx in range(num_accumulation_rounds):
            _, contexts = next(dataset_prompt_text_iterator)
            if sid_w_neg:
                contexts_neg = np.random.choice(
                    forget_dataset_prompt_text_obj.neg_prompts, (len(contexts),), replace=True).tolist()
            else:
                contexts_neg = None
            if use_context_dropout_train_fake:
                bool_tensor = torch.rand(len(contexts)) < 0.1
                # Update contexts based on bool_tensor
                contexts = ["" if flag else context for flag, context in zip(bool_tensor.tolist(), contexts)]
            z = torch.randn([len(contexts), latent_img_channels, latent_resolution, latent_resolution],
                            device=device, dtype=torch.float32)
            noise = torch.randn_like(z)

            # Initialize timesteps
            init_timesteps = init_timestep * torch.ones((len(contexts),), device=device, dtype=torch.long)

            # Generate fake images (stop generator gradient)
            with misc.ddp_sync(G_ddp, False):
                with torch.no_grad():
                    images = sid_sd_sampler(unet=G_ddp, latents=z, contexts=contexts, init_timesteps=init_timesteps,
                                            noise_scheduler=noise_scheduler,
                                            text_encoder=text_encoder, tokenizer=tokenizer,
                                            resolution=resolution, dtype=dtype, return_images=False, vae=None,
                                            num_steps=num_steps)

            timesteps = torch.randint(tmin, tmax, (len(contexts),), device=device, dtype=torch.long)

            # Compute remain loss for fake score network
            with misc.ddp_sync(fake_score_ddp, (round_idx == num_accumulation_rounds - 1)):
                # Denoised fake images (stop generator gradient) under fake score network, using guidance scale: kappa1=cfg_eval_train
                noise_fake = sid_sd_denoise(unet=fake_score_ddp, images=images, noise=noise, contexts=contexts,
                                            timesteps=timesteps,
                                            noise_scheduler=noise_scheduler,
                                            text_encoder=text_encoder, tokenizer=tokenizer,
                                            resolution=resolution, dtype=dtype, predict_x0=False,
                                            guidance_scale=cfg_train_fake,
                                            contexts_neg=contexts_neg if use_neg[0] else None)
                with torch.no_grad():
                    nan_mask = torch.isnan(noise_fake).flatten(start_dim=1).any(dim=1)
                    if noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(images, noise, timesteps)
                        nan_mask = nan_mask | torch.isnan(target).flatten(start_dim=1).any(dim=1)

                # Check if there are any NaN values present
                target = None
                if nan_mask.any():
                    # Invert the nan_mask to get a mask of samples without NaNs
                    non_nan_mask = ~nan_mask
                    # Filter out samples with NaNs from y_real and y_fake
                    noise_fake = noise_fake[non_nan_mask]
                    noise = noise[non_nan_mask]
                    if noise_scheduler.config.prediction_type == "v_prediction":
                        target = target[non_nan_mask]

                if noise_scheduler.config.prediction_type == "v_prediction":
                    sg_remain_loss = (noise_fake - target) ** 2
                    snr = compute_snr(noise_scheduler, timesteps)
                    sg_remain_loss = sg_remain_loss * snr / (snr + 1)
                else:
                    sg_remain_loss = (noise_fake - noise) ** 2

                sg_remain_loss = sg_remain_loss.sum().mul(loss_scaling / batch_gpu_total)

                if len(noise) > 0:
                    sg_remain_loss.mul(sg_remain_coef).backward()

                del images, target
                del noise_fake

                sg_remain_loss_print += sg_remain_loss.detach().item() / num_accumulation_rounds

                del sg_remain_loss

            if sg_forget_coef > 0:
                contexts_forget, contexts_neg = next(forget_dataset_prompt_text_iterator)
                if use_context_dropout_train_fake:
                    bool_tensor = torch.rand(len(contexts_forget)) < 0.1
                    # Update contexts based on bool_tensor
                    contexts_forget = ["" if flag else context_forget for flag, context_forget in
                                       zip(bool_tensor.tolist(), contexts_forget)]
                    if forget_dataset_prompt_text_obj.has_neg:
                        if sg_w_override:
                            contexts_forget, contexts_neg = list(
                            zip(*[["", ""] if flag else context_neg for flag, context_neg in
                                zip(bool_tensor.tolist(), zip(*contexts_neg))]))
                        else:
                            _, contexts_neg = list(
                                zip(*[["", ""] if flag else context_neg for flag, context_neg in
                                    zip(bool_tensor.tolist(), zip(*contexts_neg))]))
                    else:
                        contexts_neg = None
                else:
                    if forget_dataset_prompt_text_obj.has_neg:
                        _, contexts_neg = contexts_neg
                    else:
                        contexts_neg = None

                z = torch.randn([len(contexts_forget), latent_img_channels, latent_resolution, latent_resolution],
                                device=device, dtype=torch.float32)
                noise = torch.randn_like(z)

                # Initialize timesteps
                init_timesteps = init_timestep * torch.ones((len(contexts_forget),), device=device,
                                                            dtype=torch.long)

                # Generate fake images (stop generator gradient)
                with misc.ddp_sync(G_ddp, False):
                    with torch.no_grad():
                        images = sid_sd_sampler(unet=G_ddp, latents=z, contexts=contexts_forget,
                                                init_timesteps=init_timesteps,
                                                noise_scheduler=noise_scheduler,
                                                text_encoder=text_encoder, tokenizer=tokenizer,
                                                resolution=resolution, dtype=dtype, return_images=False, vae=None,
                                                num_steps=num_steps)

                timesteps = torch.randint(tmin, tmax, (len(contexts_forget),), device=device, dtype=torch.long)

                # Compute forget loss for fake score network
                with misc.ddp_sync(fake_score_ddp, (round_idx == num_accumulation_rounds - 1)):
                    # Denoised fake images (stop generator gradient) under fake score network, using guidance scale: kappa1=cfg_eval_train
                    noise_fake = sid_sd_denoise(unet=fake_score_ddp, images=images, noise=noise,
                                                contexts=contexts_forget,
                                                timesteps=timesteps,
                                                noise_scheduler=noise_scheduler,
                                                text_encoder=text_encoder, tokenizer=tokenizer,
                                                resolution=resolution, dtype=dtype, predict_x0=False,
                                                guidance_scale=cfg_train_fake,
                                                contexts_neg=contexts_neg if use_neg[0] else None,
                                                )

                    with torch.no_grad():
                        nan_mask = torch.isnan(noise_fake).flatten(start_dim=1).any(dim=1)
                        if noise_scheduler.config.prediction_type == "v_prediction":
                            target = noise_scheduler.get_velocity(images, noise, timesteps)
                            nan_mask = nan_mask | torch.isnan(target).flatten(start_dim=1).any(dim=1)

                    # Check if there are any NaN values present
                    target = None
                    if nan_mask.any():
                        # Invert the nan_mask to get a mask of samples without NaNs
                        non_nan_mask = ~nan_mask
                        # Filter out samples with NaNs from y_real and y_fake
                        noise_fake = noise_fake[non_nan_mask]
                        noise = noise[non_nan_mask]
                        if noise_scheduler.config.prediction_type == "v_prediction":
                            target = target[non_nan_mask]

                    if noise_scheduler.config.prediction_type == "v_prediction":
                        sg_forget_loss = (noise_fake - target) ** 2
                        snr = compute_snr(noise_scheduler, timesteps)
                        sg_forget_loss = sg_forget_loss * snr / (snr + 1)
                    else:
                        sg_forget_loss = (noise_fake - noise) ** 2

                    sg_forget_loss = sg_forget_loss.sum().mul(loss_scaling / batch_gpu_total)

                    if len(noise) > 0:
                        sg_forget_loss.mul(sg_forget_coef).backward()

                    del images, target
                    del noise_fake

                    sg_forget_loss = sg_forget_loss.detach().cpu().item()
                    sg_forget_loss_print += sg_forget_loss / num_accumulation_rounds

                    del sg_forget_loss

        training_stats.report('fake_score_Loss/remain_loss', sg_remain_loss_print)
        training_stats.report('fake_score_Loss/forget_loss', sg_forget_loss_print)

        fake_score_ddp.eval().requires_grad_(False)

        # Update fake score network
        for param in fake_score.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)

        fake_score_optimizer.step()

        # ----------------------------------------------------------------------------------------------
        # Update One-Step Generator Network

        G_ddp.train().requires_grad_(True)
        g_optimizer.zero_grad(set_to_none=True)

        g_remain_loss_print = g_forget_loss_print = 0
        for round_idx in range(num_accumulation_rounds):
            _, contexts = next(dataset_prompt_text_iterator)
            if sid_w_neg:
                contexts_neg = np.random.choice(
                    forget_dataset_prompt_text_obj.neg_prompts, (len(contexts),), replace=True).tolist()
            else:
                contexts_neg = None
            if use_context_dropout_train_G:
                bool_tensor = torch.rand(len(contexts)) < 0.1
                # Update contexts based on bool_tensor
                contexts = ["" if flag else context for flag, context in zip(bool_tensor.tolist(), contexts)]

            z = torch.randn([len(contexts), latent_img_channels, latent_resolution, latent_resolution],
                            device=device, dtype=torch.float32)
            noise = torch.randn_like(z)

            # initialize timesteps
            init_timesteps = init_timestep * torch.ones((len(contexts),), device=device, dtype=torch.long)
            timesteps = torch.randint(tmin, tmax, (len(contexts),), device=device, dtype=torch.long)

            # Generate fake images (track generator gradient)
            with misc.ddp_sync(G_ddp, (round_idx == num_accumulation_rounds - 1)):
                images = sid_sd_sampler(unet=G_ddp, latents=z, contexts=contexts, init_timesteps=init_timesteps,
                                        noise_scheduler=noise_scheduler,
                                        text_encoder=text_encoder, tokenizer=tokenizer,
                                        resolution=resolution, dtype=dtype, return_images=False,
                                        num_steps=num_steps)

            # Compute loss for generator
            with misc.ddp_sync(fake_score_ddp, False):
                # Denoised fake images (track generator gradient) under fake score network, using guidance scale: kappa2=kappa3=cfg_eval_fake
                y_fake = sid_sd_denoise(unet=fake_score_ddp, images=images, noise=noise, contexts=contexts,
                                        timesteps=timesteps,
                                        noise_scheduler=noise_scheduler,
                                        text_encoder=text_encoder, tokenizer=tokenizer,
                                        resolution=resolution, dtype=dtype, guidance_scale=cfg_eval_fake,
                                        contexts_neg=contexts_neg if use_neg[1] else None)

                # Denoised fake images (track generator gradient) under pretrained score network, using guidance scale: kappa4=cfg_eval_real
                y_real = sid_sd_denoise(unet=true_score, images=images, noise=noise, contexts=contexts,
                                        timesteps=timesteps,
                                        noise_scheduler=noise_scheduler,
                                        text_encoder=text_encoder, tokenizer=tokenizer,
                                        resolution=resolution, dtype=dtype, guidance_scale=cfg_eval_real,
                                        contexts_neg=contexts_neg if use_neg[2] else None)

                with torch.no_grad():
                    nan_mask_images = torch.isnan(images).flatten(start_dim=1).any(dim=1)
                    nan_mask_y_real = torch.isnan(y_real).flatten(start_dim=1).any(dim=1)
                    nan_mask_y_fake = torch.isnan(y_fake).flatten(start_dim=1).any(dim=1)
                    nan_mask = nan_mask_images | nan_mask_y_real | nan_mask_y_fake

                # Check if there are any NaN values present
                if nan_mask.any():
                    # Invert the nan_mask to get a mask of samples without NaNs
                    non_nan_mask = ~nan_mask
                    # Filter out samples with NaNs from y_real and y_fake
                    images = images[non_nan_mask]
                    y_real = y_real[non_nan_mask]
                    y_fake = y_fake[non_nan_mask]

                with torch.no_grad():
                    weight_factor = abs(images.to(torch.float32) - y_real.to(torch.float32)).mean(
                        dim=[1, 2, 3], keepdim=True).clip(min=0.00001)

                if alpha == 1:
                    g_remain_loss = (y_real - y_fake) * (y_fake - images) / weight_factor
                else:
                    g_remain_loss = (y_real - y_fake) * (
                                (y_real - images) - alpha * (y_real - y_fake)) / weight_factor

                g_remain_loss = g_remain_loss.sum().mul(loss_scaling_G / batch_gpu_total)

                if (~nan_mask).sum().item() > 0:
                    g_remain_loss.mul(g_remain_coef).backward()

                g_remain_loss = g_remain_loss.detach().cpu().item()
                g_remain_loss_print += g_remain_loss / num_accumulation_rounds

                del y_real, y_fake, images, g_remain_loss

            if g_forget_coef > 0:
                contexts_forget, contexts_override = next(forget_dataset_prompt_text_iterator)
                contexts_neg = None
                if use_context_dropout_train_G:
                    bool_tensor = torch.rand(len(contexts_forget)) < 0.1
                    # Update contexts based on bool_tensor
                    contexts_forget = ["" if flag else context_forget for flag, context_forget in
                                       zip(bool_tensor.tolist(), contexts_forget)]
                    if forget_dataset_prompt_text_obj.has_neg:
                        contexts_override, contexts_neg = list(zip(*[["", ""] if flag else context_override for flag, context_override in
                                             zip(bool_tensor.tolist(), zip(*contexts_override))]))
                    else:
                        contexts_override = ["" if flag else context_override for flag, context_override in
                                            zip(bool_tensor.tolist(), contexts_override)]
                else:
                    if forget_dataset_prompt_text_obj.has_neg:
                        contexts_override, contexts_neg = contexts_override

                z = torch.randn(
                    [len(contexts_forget), latent_img_channels, latent_resolution, latent_resolution],
                    device=device, dtype=torch.float32)
                noise = torch.randn_like(z)

                # initialize timesteps
                init_timesteps = init_timestep * torch.ones((len(contexts_forget),), device=device, dtype=torch.long)
                timesteps = torch.randint(tmin, tmax, (len(contexts_forget),), device=device, dtype=torch.long)

                # Generate fake images (track generator gradient)
                with misc.ddp_sync(G_ddp, (round_idx == num_accumulation_rounds - 1)):
                    images = sid_sd_sampler(unet=G_ddp, latents=z, contexts=contexts_forget,
                                            init_timesteps=init_timesteps,
                                            noise_scheduler=noise_scheduler,
                                            text_encoder=text_encoder, tokenizer=tokenizer,
                                            resolution=resolution, dtype=dtype, return_images=False,
                                            num_steps=num_steps)

                # Compute loss for generator
                with misc.ddp_sync(fake_score_ddp, False):
                    # Denoised fake images (track generator gradient) under fake score network, using guidance scale: kappa2=kappa3=cfg_eval_fake
                    y_fake = sid_sd_denoise(unet=fake_score_ddp, images=images, noise=noise,
                                            contexts=contexts_override if sg_w_override else contexts_forget,
                                            timesteps=timesteps,
                                            noise_scheduler=noise_scheduler,
                                            text_encoder=text_encoder, tokenizer=tokenizer,
                                            resolution=resolution, dtype=dtype, guidance_scale=cfg_eval_fake,
                                            contexts_neg=contexts_neg if use_neg[1] else None)

                    # Denoised fake images (track generator gradient) under pretrained score network, using guidance scale: kappa4=cfg_eval_real
                    y_real = sid_sd_denoise(unet=true_score, images=images, noise=noise, contexts=contexts_override,
                                            timesteps=timesteps,
                                            noise_scheduler=noise_scheduler,
                                            text_encoder=text_encoder, tokenizer=tokenizer,
                                            resolution=resolution, dtype=dtype, guidance_scale=cfg_eval_real,
                                            contexts_neg=contexts_neg if use_neg[2] else None)

                    with torch.no_grad():
                        nan_mask_images = torch.isnan(images).flatten(start_dim=1).any(dim=1)
                        nan_mask_y_real = torch.isnan(y_real).flatten(start_dim=1).any(dim=1)
                        nan_mask_y_fake = torch.isnan(y_fake).flatten(start_dim=1).any(dim=1)
                        nan_mask = nan_mask_images | nan_mask_y_real | nan_mask_y_fake

                    # Check if there are any NaN values present
                    if nan_mask.any():
                        # Invert the nan_mask to get a mask of samples without NaNs
                        non_nan_mask = ~nan_mask
                        # Filter out samples with NaNs from y_real and y_fake
                        images = images[non_nan_mask]
                        y_real = y_real[non_nan_mask]
                        y_fake = y_fake[non_nan_mask]

                    with torch.no_grad():
                        weight_factor = abs(images.to(torch.float32) - y_real.to(torch.float32)).mean(
                            dim=[1, 2, 3], keepdim=True).clip(min=0.00001)

                    if alpha == 1:
                        g_forget_loss = (y_real - y_fake) * (y_fake - images) / weight_factor
                    else:
                        g_forget_loss = (y_real - y_fake) * (
                                    (y_real - images) - alpha * (y_real - y_fake)) / weight_factor

                    g_forget_loss = g_forget_loss.sum().mul(loss_scaling_G / batch_gpu_total)

                    if (~nan_mask).sum().item() > 0:
                        g_forget_loss.mul(g_forget_coef).backward()

                    g_forget_loss = g_forget_loss.detach().cpu().item()
                    g_forget_loss_print += g_forget_loss / num_accumulation_rounds

                    del y_real, y_fake, images, g_forget_loss

        training_stats.report('G_Loss/remain_loss', g_remain_loss_print)
        training_stats.report('G_Loss/forget_loss', g_forget_loss_print)

        G_ddp.eval().requires_grad_(False)

        # Update generator
        for param in G.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)

        # Apply gradient clipping under fp16 to prevent suddern divergence
        if dtype == torch.float16 and (~nan_mask).sum().item() > 0:
            torch.nn.utils.clip_grad_value_(G.parameters(), 1)

        g_optimizer.step()

        if ema_halflife_kimg > 0:
            # Update EMA.
            ema_halflife_nimg = ema_halflife_kimg * 1000
            if ema_rampup_ratio is not None:
                ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
            ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))

            for p_ema, p_true_score in zip(G_ema.parameters(), G.parameters()):
                with torch.no_grad():
                    p_ema.copy_(p_true_score.detach().lerp(p_ema, ema_beta))
        else:
            G_ema = G

        torch.cuda.empty_cache()
        gc.collect()

        cur_nimg += batch_size
        done = (cur_nimg >= total_kimg * 1000)

        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
        fields += [
            f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [
            f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [
            f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [
            f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2 ** 30):<6.2f}"]
        fields += [
            f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2 ** 30):<6.2f}"]
        fields += [
            f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2 ** 30):<6.2f}"]
        fields += [
            f"loss_fake_score_remain {training_stats.report0('fake_score_Loss/remain_loss', sg_remain_loss_print):<6.2f}"]
        fields += [
            f"loss_fake_score_forget {training_stats.report0('fake_score_Loss/forget_loss', sg_forget_loss_print):<6.2f}"]
        fields += [f"loss_G_remain {training_stats.report0('G_Loss/remain_loss', g_remain_loss_print):<6.2f}"]
        fields += [f"loss_G_forget {training_stats.report0('G_Loss/forget_loss', g_forget_loss_print):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        dist.print0(' '.join(fields))

        # Check for abort.
        if (not done) and dist.should_stop():
            done = True
            dist.print0()
            dist.print0('Aborting...')

        if (snapshot_ticks is not None) and (
                done or cur_tick % snapshot_ticks == 0 or cur_tick in [2, 4, 10, 20, 30, 40, 50, 60, 70, 80, 90,
                                                                       100]):

            dist.print0('Exporting sample images...')
            if dist.get_rank() == 0:
                for num_steps_eval in [1, 2, 4]:
                    # While the generator is primarily trained to generate images in a single step, it can also be utilized in a multi-step setting during evaluation.
                    # To do: Distill a multi-step generator that is optimized for multi-step settings
                    with torch.no_grad():
                        images = [sid_sd_sampler(unet=G_ema, latents=z, contexts=c,
                                                 init_timesteps=init_timestep * torch.ones(
                                                     (len(c),), device=device, dtype=torch.long),
                                                 noise_scheduler=noise_scheduler,
                                                 text_encoder=text_encoder, tokenizer=tokenizer,
                                                 resolution=resolution, dtype=dtype, return_images=True, vae=vae,
                                                 num_steps=num_steps, train_sampler=False,
                                                 num_steps_eval=num_steps_eval).cpu() for z, c in zip(grid_z, grid_c)]
                    images = torch.cat(images).cpu().numpy()

                    save_image_grid(img=images, fname=os.path.join(
                        run_dir, f'fakes_{alpha:03f}_{cur_nimg // 1000:06d}_{num_steps_eval:d}.png'),
                        drange=[-1, 1], grid_size=grid_size)

                del images

            data = dict(ema=G_ema)
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    from collections import OrderedDict

                    value_state_dict = OrderedDict([(k, v.detach().cpu()) for k, v in value.state_dict().items()])
                    unet_cpu_copy.load_state_dict(value_state_dict)
                    data[key] = unet_cpu_copy
                    del value_state_dict

            if dist.get_rank() == 0:
                save_data(data=data,
                          fname=os.path.join(run_dir, f'network-snapshot-{alpha:03f}-{cur_nimg // 1000:06d}.pkl'))

            del data  # conserve memory

        if (state_dump_ticks is not None) and (
                done or cur_tick % state_dump_ticks == 0) and cur_tick != 0 and dist.get_rank() == 0:
            dist.print0(f'saving checkpoint: training-state-{cur_nimg // 1000:06d}.pt')
            save_pt(pt=dict(fake_score=fake_score, G=G, G_ema=G_ema,
                            fake_score_optimizer_state=fake_score_optimizer.state_dict(),
                            g_optimizer_state=g_optimizer.state_dict()),
                    fname=os.path.join(run_dir, f'training-state-{cur_nimg // 1000:06d}.pt'))

        # Update logs.
        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            if stats_jsonl is None:
                append_line(jsonl_line=json.dumps(
                    dict(training_stats.default_collector.as_dict(), timestamp=time.time())) + '\n',
                            fname=os.path.join(run_dir, f'stats_{alpha:03f}.jsonl'))

        dist.update_progress(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    dist.print0()
    dist.print0('Exiting...')

# ----------------------------------------------------------------------------