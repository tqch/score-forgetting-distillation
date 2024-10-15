import argparse
import logging
import os
import sys
import traceback

import numpy as np
import torch
import yaml
from functions import dict2namespace
from runners.diffusion import Diffusion

torch.set_printoptions(sci_mode=False)


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument(
        "--ckpt_folder",
        type=str,
        help="Path to folder with pretrained model for sampling (only necessary if sampling)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["sample_fid", "sample_classes", "visualization"],
        help="Sampling mode.",
    )
    parser.add_argument(
        "--n_samples_per_class",
        type=int,
        default=5000,
        help="Number of samples per class to generate.",
    )
    parser.add_argument(
        "--classes_to_generate",
        type=str,
        default="0,1,2,3,4,5,6,7,8,9",
        help="Either a comma-separated string of class labels to generate e.g, '0,1,2,3', \
            otherwise prefix 'x' to drop that class , e.g., 'x0, x1' to generate all classes but 0 and 1.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument(
        "--sample_type",
        type=str,
        default="generalized",
        help="sampling approach (generalized or ddpm_noisy)",
    )
    parser.add_argument(
        "--skip_type",
        type=str,
        default="uniform",
        help="skip according to (uniform or quadratic)",
    )
    parser.add_argument(
        "--timesteps", type=int, default=1000, help="number of steps involved"
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=1.0,
        help="eta used to control the variances of sigma",
    )
    parser.add_argument(
        "--cond_scale",
        type=float,
        default=2.0,
        help="classifier-free guidance conditional strength",
    )
    parser.add_argument("--sequence", action="store_true")
    parser.add_argument("--no_ema", action="store_true")
    parser.add_argument("--start_img_id", type=int, required=False)

    args = parser.parse_args()

    config = os.path.join("configs", args.config)
    if not os.path.exists(config):
        config = args.config
    with open(config, "r") as fp:
        config = yaml.unsafe_load(fp)
        if isinstance(config, argparse.Namespace):
            config = dict(config._get_kwargs())
        config = dict2namespace(config)

    if args.no_ema:
        config.model.ema = False

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, config


def main():
    import re

    args, config = parse_args_and_config()

    dataset_name = config.data.dataset.lower()
    try:
        date_time = re.search(r"\d{4}_\d{2}_\d{2}_\d{6}", args.ckpt_folder).group(0)
    except AttributeError:
        date_time = "unknown"
    if args.mode == "visualization":
        config.log_dir = f"./results/{dataset_name}/visualization/{date_time}"
        os.makedirs(config.log_dir, exist_ok=True)

    try:
        runner = Diffusion(args, config)
        runner.sample()
    except Exception:
        logging.error(traceback.format_exc())

    return 0


if __name__ == "__main__":
    sys.exit(main())

# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6" python sample.py --config cifar10_sample.yml --ckpt_folder results/cifar10/2023_08_16_224303 --mode sample_classes --n_samples_per_class 500
# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6" python fim.py --config cifar10_fim.yml --ckpt_folder results/cifar10/2023_08_16_224303 --n_chunks 20

# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python sample.py --config cifar10_sample.yml --ckpt_folder results/cifar10/2023_08_16_224303 --mode sample_fid --n_samples_per_class 500 --classes_to_generate 'x0'
# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python sample.py --config cifar10_sample.yml --ckpt_folder results/cifar10/2023_08_16_224303 --mode sample_classes --classes_to_generate "0" --n_samples_per_class 500

# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python evaluator.py results/cifar10/2023_08_16_224303/fid_samples_guidance_2.0_excluded_class_0 cifar10_without_label_0

# python sample.py --config cifar10_sample.yml --ckpt_folder results/cifar10/2023_08_16_224303 --mode visualization --cond_scale -1
# python sample.py --config cifar10_sample.yml --ckpt_folder results/cifar10/2023_08_28_232149 --mode visualization --cond_scale -1


# CUDA_VISIBLE_DEVICES="0,1" python sample.py --config stl10_sample.yml --ckpt_folder results/cifar10/2024_05_12_224610 --mode sample_fid --n_samples_per_class 5000 --classes_to_generate 'x0'

# CUDA_VISIBLE_DEVICES="0,1" python sample.py --config stl10_sample.yml --ckpt_folder results/stl10/2024_05_21_101737/ckpts/ckpt-24999.pth --mode visualization --timesteps 50 --eta 0 --no_ema

# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python sample.py --config cifar10_sample.yml --ckpt_folder results/cifar10/2024_06_03_023925 --mode sample_fid --n_samples_per_class 5000 --classes_to_generate '0,1,2,3,4,5,6,7,8,9' --cond_scale 0.0
# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python sample.py --config cifar10_sample.yml --ckpt_folder results/cifar10/2024_06_03_023925 --mode sample_fid --n_samples_per_class 5000 --classes_to_generate '0,1,2,3,4,5,6,7,8,9' --cond_scale 2.0

# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python sample.py --config cifar10_sample.yml --ckpt_folder results/cifar10/2024_06_08_172724 --mode sample_fid --n_samples_per_class 5000 --classes_to_generate '0,1,2,3,4,5,6,7,8,9' --cond_scale 0.0
# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python sample.py --config cifar10_sample.yml --ckpt_folder results/cifar10/2024_06_08_172724 --mode sample_fid --n_samples_per_class 5000 --classes_to_generate '0,1,2,3,4,5,6,7,8,9' --cond_scale 2.0

# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python sample.py --config cifar10_sample.yml --ckpt_folder results/cifar10/2024_06_11_004401 --mode sample_fid --n_samples_per_class 5000 --classes_to_generate '0,1,2,3,4,5,6,7,8,9' --cond_scale 0.0
# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python sample.py --config cifar10_sample.yml --ckpt_folder results/cifar10/2024_06_11_004401 --mode sample_fid --n_samples_per_class 5000 --classes_to_generate '0,1,2,3,4,5,6,7,8,9' --cond_scale 2.0

# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python sample.py --config results/cifar10/2024_07_15_221705/logs/config.yaml --ckpt_folder results/cifar10/2024_07_15_221705 --mode sample_fid --n_samples_per_class 5000 --classes_to_generate '0,1,2,3,4,5,6,7,8,9' --cond_scale 0.0
# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python sample.py --config results/cifar10/2024_07_21_183729/logs/config.yaml --ckpt_folder results/cifar10/2024_07_21_183729 --mode sample_fid --n_samples_per_class 5000 --classes_to_generate '0,1,2,3,4,5,6,7,8,9' --cond_scale 0.0

# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python sample.py --config results/cifar10/forget/rl/0.001_full/2024_09_28_131547/logs/config.yaml --ckpt_folder results/cifar10/forget/rl/0.001_full/2024_09_28_131547/ckpts/ckpt-999.pth --mode sample_fid --n_samples_per_class 5000 --classes_to_generate '0,1,2,3,4,5,6,7,8,9'

# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python sample.py --config results/stl10/2024_07_14_102349/logs/config.yaml --ckpt_folder results/stl10/2024_07_14_102349/ckpts/ckpt-999.pth --mode sample_fid --n_samples_per_class 5000 --classes_to_generate '0,1,2,3,4,5,6,7,8,9' --sample_type one_step

# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python sample.py --config results/stl10/2024_08_03_021143/logs/config.yaml --ckpt_folder results/stl10/2024_08_03_021143/ckpts/ckpt-49999.pth --mode sample_fid --n_samples_per_class 5000 --classes_to_generate '0,1,2,3,4,5,6,7,8,9' --sample_type one_step --cond_scale 0.0

# SAMPLE_DIR_SUFFIX="_4999" CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python sample.py --config results/stl10/2024_07_14_102349/logs/config.yaml --ckpt_folder results/stl10/2024_07_14_102349/ckpts/ckpt-4999.pth --mode sample_fid --n_samples_per_class 50 --classes_to_generate '0,1,2,3,4,5,6,7,8,9' --sample_type one_step --cond_scale 0.0
# SAMPLE_DIR_SUFFIX="_9999" CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python sample.py --config results/stl10/2024_07_14_102349/logs/config.yaml --ckpt_folder results/stl10/2024_07_14_102349/ckpts/ckpt-9999.pth --mode sample_fid --n_samples_per_class 50 --classes_to_generate '0,1,2,3,4,5,6,7,8,9' --sample_type one_step --cond_scale 0.0
# SAMPLE_DIR_SUFFIX="_19999" CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python sample.py --config results/stl10/2024_07_14_102349/logs/config.yaml --ckpt_folder results/stl10/2024_07_14_102349/ckpts/ckpt-19999.pth --mode sample_fid --n_samples_per_class 50 --classes_to_generate '0,1,2,3,4,5,6,7,8,9' --sample_type one_step --cond_scale 0.0
# SAMPLE_DIR_SUFFIX="_29999" CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python sample.py --config results/stl10/2024_07_14_102349/logs/config.yaml --ckpt_folder results/stl10/2024_07_14_102349/ckpts/ckpt-29999.pth --mode sample_fid --n_samples_per_class 50 --classes_to_generate '0,1,2,3,4,5,6,7,8,9' --sample_type one_step --cond_scale 0.0
# SAMPLE_DIR_SUFFIX="_49999" CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python sample.py --config results/stl10/2024_07_14_102349/logs/config.yaml --ckpt_folder results/stl10/2024_07_14_102349/ckpts/ckpt-49999.pth --mode sample_fid --n_samples_per_class 50 --classes_to_generate '0,1,2,3,4,5,6,7,8,9' --sample_type one_step --cond_scale 0.0


from PIL import Image
import numpy as np
import glob
import os
folders = glob.glob('results/stl10/2024_07_14_102349/fid_samples_guidance_0.0_*')
sel_imgs = [
    '001_00010',
    '002_00012',
    '003_00020',
    '004_00021',
    '005_00048',
    '006_00037',
    '007_00036',
    '008_00030',
    '009_00017',
]
for folder in folders:
    steps = folder.rsplit('_', maxsplit=1)[1]
    Image.fromarray(
        np.stack([np.array(Image.open(os.path.join(folder, sel_img + '.png'))) for sel_img in sel_imgs], axis=0) \
        .reshape(3, 3, 64, 64, 3).transpose(0, 2, 1, 3, 4).reshape(192, 192, 3)
    ).resize((320, 320), resample=Image.LANCZOS).save(os.path.join(
        os.path.dirname(folder), f'STL10_{steps}_3x3.png'))
