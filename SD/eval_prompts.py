import os
import re
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
from torch_utils import distributed as dist
from functools import partial
from training.sd_util import sid_sd_sampler
from diffusers import DiffusionPipeline
from diffusers import DDPMScheduler

import math
#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges


@click.command()
@click.option('--network', 'network_pkl',  help='Network pickle filename', metavar='PATH|URL',                      type=str, required=True)
@click.option('--outdir',                  help='Where to save the output images', metavar='DIR',                   type=str, required=True)
@click.option('--seeds',                   help='Random seeds (e.g. 1,2,5-10)', metavar='LIST',                     type=parse_int_list, default='0-799', show_default=True)
@click.option('--subdirs',                 help='Create subdirectory for every 1000 seeds',                         is_flag=True)
@click.option('--batch', 'max_batch_size', help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=64, show_default=True)
# @click.option('--num', 'num_fid_samples',  help='Maximum num of images', metavar='INT',                             type=click.IntRange(min=1), default=50000, show_default=True)
@click.option('--init_timestep', 'init_timestep',      help='Stoch. noise std', metavar='long',                      type=int, default=625, show_default=True)
@click.option('--repo_id', 'repo_id',   help='diffusion pipeline filename', metavar='PATH|URL',         type=str, default='runwayml/stable-diffusion-v1-5', show_default=True)
@click.option('--forget_data_prompt_text', metavar='PATH', type=str)
@click.option('--forget_data_prompt_text_val', metavar='PATH', type=str)
@click.option('--pos_data_prompt_text', metavar='PATH', type=str)
@click.option('--plot_grid', metavar='INT', type=int, default=0, show_default=True)
@click.option('--resolution', metavar='INT', type=int, default=512, show_default=True)
@click.option('--numpy_seed', metavar='INT', type=int, default=0, show_default=True)

def main(network_pkl, outdir, seeds, subdirs, max_batch_size, init_timestep, repo_id, forget_data_prompt_text, forget_data_prompt_text_val, pos_data_prompt_text, plot_grid, resolution, numpy_seed):
    dist.init()
    dtype = torch.float16
    device = "cuda"

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load network.
    dist.print0(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
        G_ema = pickle.load(f)['ema'].to(device).to(dtype)
        # e.g., network-snapshot-1.000000-000000
        try:
            match = re.match('network-snapshot-([.0-9]{7,})-([0-9]{6,})\.pkl', os.path.basename(network_pkl))
            alpha = float(match.group(1))
            cur_nimg = int(match.group(2)) * 1000
        except AttributeError:
            alpha = 1.0
            cur_nimg = 0

    pipeline = DiffusionPipeline.from_pretrained(repo_id, use_safetensors=True)

    tokenizer = pipeline.tokenizer
    text_encoder = pipeline.text_encoder.to(device, dtype)
    vae = pipeline.vae.to(device, dtype)
    noise_scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)

    num_steps = 1
    num_steps_eval = 1
    G = partial(sid_sd_sampler, unet=G_ema, noise_scheduler=noise_scheduler,
                text_encoder=text_encoder, tokenizer=tokenizer,
                resolution=resolution, dtype=dtype, return_images=True, vae=vae, num_steps=num_steps, train_sampler=False,
                num_steps_eval=num_steps_eval)

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    extra_kwargs = dict()
    if plot_grid:
        extra_kwargs["max_size"] = 8

    dataset_kwargs = dnnlib.EasyDict(
        class_name='training.dataset.ForgetPromptDataset', resolution=512,
        path=forget_data_prompt_text, random_seed=None,
        concept_to_forget="", concept_to_override="",
        path_to_val=forget_data_prompt_text_val, path_to_pos=pos_data_prompt_text, path_to_neg=None,
        **extra_kwargs
    )
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)

    if plot_grid:
        batch_gpu = max_batch_size
        from training.training_loop import setup_snapshot_image_grid, split_list, save_image_grid
        # Parameters for latent diffusion
        latent_img_channels = 4
        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        latent_resolution = resolution // vae_scale_factor
        seed = numpy_seed
        np.random.seed((seed * dist.get_world_size() + 0 * dist.get_rank()) % (1 << 31))  # always using rank 0
        grid_size, _, contexts = setup_snapshot_image_grid(training_set=dataset_obj, random_seed=None)
        grid_z = torch.randn([
            len(contexts), latent_img_channels, latent_resolution, latent_resolution], device=device, dtype=dtype,
            generator=torch.Generator(device=device).manual_seed(2024)
        )
        grid_z = grid_z.split(batch_gpu)
        grid_c = split_list(contexts, batch_gpu)

        local_grid_z = grid_z[dist.get_rank()::dist.get_world_size()]
        local_grid_c = grid_c[dist.get_rank()::dist.get_world_size()]

        with torch.no_grad():
            local_images = torch.cat([sid_sd_sampler(unet=G_ema, latents=z, contexts=c,
                                     init_timesteps=init_timestep * torch.ones(
                                         (len(c),), device=device, dtype=torch.long),
                                     noise_scheduler=noise_scheduler,
                                     text_encoder=text_encoder, tokenizer=tokenizer,
                                     resolution=resolution, dtype=dtype, return_images=True, vae=vae,
                                     num_steps=num_steps, train_sampler=False,
                                     num_steps_eval=num_steps_eval) for z, c in zip(local_grid_z, local_grid_c)])
        torch.distributed.barrier()

        if dist.get_world_size() > 1:
            images = []
            for src in range(dist.get_world_size()):
                _local_images = local_images.clone()
                torch.distributed.broadcast(_local_images, src=src)
                images.append(_local_images.cpu())
                del _local_images
            images = torch.cat(images, dim=0).reshape(
                dist.get_world_size(), -1, *images[0].shape
            ).permute(1, 0, 2, 3, 4, 5).reshape(-1, *images[0].shape[1:]).numpy()

            if dist.get_rank() == 0:
                os.makedirs(outdir, exist_ok=True)
                save_image_grid(img=images, fname=os.path.join(
                    outdir, f'fakes_{alpha:03f}_{cur_nimg // 1000:06d}_{num_steps_eval:d}.png'),
                                drange=[-1, 1], grid_size=grid_size)

    else:
        for (prompt, _) in tqdm.tqdm(dataset_obj.prompts, unit='prompt', disable=(dist.get_rank() != 0)):

            num_batches = math.ceil(len(seeds) / max_batch_size)
            all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
            rank_batches = all_batches[dist.get_rank()::dist.get_world_size()]

            # Loop over batches.
            dist.print0(f'Generating {len(seeds)} images to "{outdir}/{prompt}"...')
            for batch_seeds in tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0)):
                torch.distributed.barrier()
                batch_size = len(batch_seeds)
                if batch_size == 0:
                    continue

                # Pick latents and labels.
                rnd = StackedRandomGenerator(device, batch_seeds)
                img_channels = 4
                img_resolution = 64
                latents = rnd.randn([batch_size, img_channels, img_resolution, img_resolution], device=device)

                c = [prompt for _ in batch_seeds]
                with torch.no_grad():
                    images = G(latents=latents, contexts=c,
                               init_timesteps=init_timestep * torch.ones((len(c),), device=latents.device, dtype=torch.long))

                # Save images.
                images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
                for seed, image_np in zip(batch_seeds, images_np):
                    image_dir = os.path.join(outdir, prompt, f'{seed - seed % 1000:06d}') if subdirs else os.path.join(outdir, prompt)
                    os.makedirs(image_dir, exist_ok=True)
                    image_path = os.path.join(image_dir, f'{seed:05d}.jpg')
                    if image_np.shape[2] == 1:
                        PIL.Image.fromarray(image_np[:, :, 0], 'L').save(image_path)
                    else:
                        PIL.Image.fromarray(image_np, 'RGB').save(image_path)

            torch.distributed.barrier()

    # Done.
    torch.distributed.barrier()
    dist.print0('Done.')

if __name__ == "__main__":
    main()
