# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import re
import json
import click
import torch
import dnnlib

from torch_utils import distributed as dist
import glob

from metrics import metric_main as metric_main
from metrics.metric_utils import ProgressMonitor
from collections import defaultdict

import warnings
from omegaconf import OmegaConf
import yaml


warnings.filterwarnings('ignore','Grad strides do not match bucket view strides')  # False warning printed by PyTorch 1.12.


# ----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges


class CommaSeparatedList(click.ParamType):
    name = 'list'

    def convert(self, value, param, ctx):
        _ = param, ctx
        if value is None or value.lower() == 'none' or value == '':
            return []
        return value.split(',')


def calculate_metric(metric, G, init_t, init_a, init_b, init_sigma, dataset_kwargs, num_gpus, rank, device, data_stat, G_kwargs=None, progress=None):
    return metric_main.calc_metric(metric=metric, G=G, init_t=init_t, init_a=init_a, init_b=init_b, init_sigma=init_sigma,
                                   dataset_kwargs=dataset_kwargs, num_gpus=num_gpus, rank=rank, device=device,
                                   data_stat=data_stat, G_kwargs=G_kwargs, progress=progress)


def append_line(jsonl_line, fname):
    with open(fname, 'at') as f:
        f.write(jsonl_line + '\n')


def save_metric(result_dict, fname):
    with open(fname, "w") as file:
        for key, value in result_dict.items():
            file.write(f"{key}: {value}\n")


# ----------------------------------------------------------------------------


def bool_int(interger):
    return bool(int(interger))


@click.command()
@click.option('--data', help='Path to the dataset', metavar='ZIP|DIR', type=str, required=True)
@click.option('--data_stat', help='Path to the dataset stats', metavar='ZIP|DIR', type=str, default=None)
@click.option('--beta_start', type=click.FloatRange(min=0, min_open=True), default=0.0001, show_default=True)
@click.option('--beta_end', type=click.FloatRange(min=0, min_open=True), default=0.02, show_default=True)
@click.option('--timesteps', type=click.IntRange(min=0, min_open=True), default=1000, show_default=True)
@click.option('--init_t', help='Initial time that is fixed during distillation and generation',
              metavar='FLOAT', type=click.IntRange(min=0, min_open=True), default=440, show_default=True)
@click.option('--cond', help='Train class-conditional model', metavar='BOOL', type=bool, default=False,
              show_default=True)
# Main options.gpu
# Adapted from Diff-Instruct
@click.option('--metrics', help='Comma-separated list or "none" [default: fid50k_full]', type=CommaSeparatedList())
# FID metric PT path
@click.option('--network', 'network_pth', help='Network pickle filename', metavar='PATH|URL', type=str, required=True)
@click.option('--network_cfg', 'network_cfg', help='Network pickle config filename', metavar='PATH|URL', type=str, required=True)
@click.option('--sample', 'sample_folder', help='Generated image folder', metavar='PATH|URL', type=str)
@click.option('--exclude', help='filename pattern to exclude', metavar='STR', type=str)
@click.option('--res', 'resolution', help='Generated image resolution', metavar='INT', type=int)
@click.option('--remain', 'is_remain', help='Whether to calculate metrics for remaining only', metavar='BOOL',
              type=bool_int, default=False, show_default=True)
@click.option('--ema', help='Whether to use ema copy', metavar='BOOL', type=bool_int, default=True, show_default=True)
@click.option('--label_to_forget', 'label_to_forget', help='label to forget', metavar='INT',
              type=int)
@click.option('--verbose', metavar='BOOL', type=bool, default=False, show_default=True)
def main(**kwargs):
    opts = dnnlib.EasyDict(kwargs)
    torch.multiprocessing.set_start_method('spawn')
    dist.init()

    result_name = "metrics"
    label_to_forget = opts.label_to_forget
    if label_to_forget is not None:
        result_name += f"_forget_{label_to_forget}"
    is_remain = opts.is_remain and label_to_forget is not None
    if is_remain:
        result_name += "_remain"
    if not opts.ema:
        result_name += "_noema"

    network_pths = opts.network_pth
    if os.path.isdir(network_pths):
        save_path = os.path.join(network_pths, f"{result_name}.json")
        network_pths = glob.glob(f"{network_pths}/ckpt-*.pth")
    else:
        assert network_pths.endswith(".pth")
        save_path = os.path.dirname(network_pths).rstrip("\/") + f"/{result_name}.json"
        network_pths = [network_pths]

    try:
        network_pths = sorted([(
            os.path.basename(network_pth).split("-", maxsplit=3)[-1].split(".")[0],
            network_pth) for network_pth in network_pths], key=lambda x: int(x[0]))
    except:
        network_pths = sorted([(
            os.path.basename(network_pth), network_pth) for network_pth in network_pths])

    metrics = opts.metrics
    init_t = opts.init_t
    data_stat = opts.data_stat
    betas = torch.linspace(opts.beta_start, opts.beta_end, opts.timesteps)
    init_a = torch.cumprod(1 - betas, dim=0)[init_t]
    init_b = (1 - init_a).sqrt().item()
    init_a = init_a.sqrt().item()
    init_sigma = init_b / init_a
    init_kwargs = dict(init_t=init_t, init_a=init_a, init_b=init_b, init_sigma=init_sigma)
    dist.print0(json.dumps(init_kwargs, indent=2))
    dist.print0(f"label_to_forget: {opts.label_to_forget}")

    torch.cuda.set_device(dist.get_rank())
    device = torch.device('cuda')

    results = defaultdict(dict)

    try:
        config = OmegaConf.load(opts.network_cfg)
    except yaml.constructor.ConstructorError:
        from io import StringIO
        with open(opts.network_cfg) as f:
            config_str = f.read().replace("!!python/object:argparse.Namespace", "")
            config = OmegaConf.load(StringIO(config_str))

    for kimgs, network_pth in network_pths:

        # Rank 0 goes first.
        if dist.get_rank() != 0:
            torch.distributed.barrier()

        if opts.sample_folder is None:
            # Load network.
            dist.print0(f'Loading network from "{network_pth}"...')

            G = dnnlib.util.construct_class_by_name(
                class_name="models.diffusion.Conditional_Model", config=config)
            G.img_channels = 3
            G.img_resolution = opts.resolution
            state_dict = torch.load(network_pth, map_location="cpu")
            if "null_classes_emb" not in state_dict[0 if len(state_dict) <= 4 else 2]:
                del G.null_classes_emb
            if opts.ema:
                G.load_state_dict(state_dict[-1], strict=True)
            else:
                assert len(state_dict) >= 5
                G.load_state_dict(state_dict[2], strict=True)
            G.eval().requires_grad_(False).to(device)

            dist.print0(f'Finished loading "{network_pth}"...')

            _dataset_kwargs = dict()
        else:
            G = None
            _dataset_kwargs = dnnlib.EasyDict(
                class_name='datasets.sid_dataset.ImageFolderDataset', path=opts.sample_folder,
                exclude=opts.exclude, resolution=opts.resolution, max_size=2000000
            )

        # Other ranks follow.
        if dist.get_rank() == 0:
            torch.distributed.barrier()

        dataset_kwargs = dnnlib.EasyDict(
            class_name='datasets.sid_dataset.ForgetImageFolderDataset',
            path=opts.data, use_labels=opts.cond,
            is_remain=is_remain, label_to_forget=opts.label_to_forget)

        dataset_kwargs.resolution = opts.resolution
        dataset_kwargs.max_size = 2000000
        dist.print0(dataset_kwargs)
        if len(_dataset_kwargs):
            dist.print0(_dataset_kwargs)

        match = re.search(r"-(\d+)\.pkl$", network_pth)
        if match:
            # If a match is found, extract the number part
            number_part = match.group(1)
        else:
            # If no match is found, handle the situation (e.g., use a default value or skip processing)
            number_part = '_final'  # Or any other handling logic you prefer
        for metric in metrics:
            dist.print0(metric)
            progress = ProgressMonitor(verbose=opts.verbose) if dist.get_rank() == 0 else None
            result_dict = calculate_metric(
                metric=metric, G=G, **init_kwargs, dataset_kwargs=dataset_kwargs,
                num_gpus=dist.get_world_size(), progress=progress,
                rank=dist.get_rank(), device=device, data_stat=data_stat, G_kwargs=_dataset_kwargs)
            if dist.get_rank() == 0:
                print(result_dict.results)
                results[kimgs][metric] = result_dict
            torch.distributed.barrier()

    if dist.get_rank() == 0:
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)

    torch.distributed.barrier()

# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------
