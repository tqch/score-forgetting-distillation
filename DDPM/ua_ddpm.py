from argparse import ArgumentParser
import glob
import pickle
import os
import re
import torch.nn as nn
import torchvision
from torchvision import transforms
import torch
import json
import yaml
from tqdm import tqdm
from collections import defaultdict
from PIL import Image
from omegaconf import OmegaConf
from models.diffusion import Conditional_Model
from runners.diffusion import get_beta_schedule
from io import StringIO

parser = ArgumentParser()
parser.add_argument("--ckpt_folder", type=str, required=True)
parser.add_argument("--clf_path", type=str, required=True)
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--n_total", type=int)
parser.add_argument("--ema", type=int, default=1)
args = parser.parse_args()

if os.path.isfile(args.ckpt_folder):
    network_pkls = [args.ckpt_folder]
else:
    network_pkls = glob.glob(os.path.join(args.ckpt_folder, "**/ckpt-*.pth"), recursive=True)
    network_pkls = sorted(network_pkls, key=lambda x: int(re.search("ckpt-(\d+).pth", x).group(1)))

device = "cuda"

model = torchvision.models.resnet34(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)
model.load_state_dict(torch.load(args.clf_path, map_location="cpu"))
model = model.eval().requires_grad_(False).to(device)

batch_size = 100
# init_sigma = 2.5
label_dim = 10
label_of_forgotten_class = 0

transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

results = defaultdict(dict)

try:
    config = OmegaConf.load(args.config)
except yaml.constructor.ConstructorError:
    with open(args.config) as f:
        config = StringIO(f.read().replace("!!python/object:argparse.Namespace", ""))
    config = OmegaConf.load(config)

img_size = config.data.image_size
n_total = args.n_total
if n_total is None:
    if config.data.dataset == "CIFAR10":
        n_total = 5000
    elif config.data.dataset == "STL10":
        n_total = 1300
shape = [3, img_size, img_size]

t_init = config.distill.t_init
betas = get_beta_schedule(
    beta_schedule=config.diffusion.beta_schedule,
    beta_start=config.diffusion.beta_start,
    beta_end=config.diffusion.beta_end,
    num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
)
betas = torch.from_numpy(betas).float().to(device)
init_a = (1 - betas).cumprod(dim=0).sqrt()[t_init].to(device)
init_b = (1 - (1 - betas).cumprod(dim=0)).sqrt()[t_init].to(device)
init_sigma = (1 / (1 - betas).cumprod(dim=0) - 1.).sqrt()[t_init].to(device)

for network_pkl in network_pkls:
    steps = os.path.basename(network_pkl).split("-", maxsplit=1)[-1].split(".")[0]
    G = Conditional_Model(config).eval().requires_grad_(False)
    if args.ema:
        state_dict = torch.load(network_pkl, map_location="cpu")[-1]
    else:
        state_dict = torch.load(network_pkl, map_location="cpu")
        if len(state_dict) > 4:
            state_dict = state_dict[2]
        else:
            state_dict = state_dict[0]
    for k in list(state_dict.keys()):
        if k.startswith("module."):
            state_dict[k[7:]] = state_dict.pop(k)
    G.load_state_dict(state_dict)
    G.to(device)

    def x0_from_noise(xt, et):
        return xt / init_a - init_sigma * et

    @torch.no_grad()
    def run_generator(noise, c):
        t = torch.empty(noise.size(0), device=device).fill_(t_init)
        xt = init_b * noise
        et = G(xt, t, c, mode="test", cond_scale=0.0, cond_drop_prob=0.0, cond_drop_mask=None)
        x0 = x0_from_noise(xt, et).mul_(127.5).add_(128).clamp_(0, 255)
        return x0.to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()

    class_labels = torch.zeros([batch_size, ], device=device, dtype=torch.int)

    entropy_cum_sum = 0
    forgotten_prob_cum_sum = 0
    accuracy_cum_sum = 0

    for _ in tqdm(range(n_total // batch_size)):
        latents = torch.randn([batch_size, ] + shape, device=device)
        with torch.no_grad():
            data = run_generator(latents, class_labels)
            data = torch.as_tensor(torch.stack([transform(Image.fromarray(d)) for d in data], dim=0))
            logits = model(data.to(device))

        pred = torch.argmax(logits, dim=-1)
        accuracy = (pred == label_of_forgotten_class).sum().item()
        accuracy_cum_sum += accuracy / n_total

        probs = torch.nn.functional.softmax(logits, dim=-1)
        log_probs = torch.log(probs)
        entropy = -torch.multiply(probs, log_probs).sum(1)
        avg_entropy = torch.sum(entropy) / n_total
        entropy_cum_sum += avg_entropy.item()
        forgotten_prob_cum_sum += (probs[:, label_of_forgotten_class] / n_total).sum().item()

    print(f"Average entropy: {entropy_cum_sum}")
    print(f"Average prob of forgotten class: {forgotten_prob_cum_sum}")
    print(f"Average accuracy of forgotten class: {accuracy_cum_sum}")

    results[steps]["entropy"] = float(entropy_cum_sum)
    results[steps]["prob"] = float(forgotten_prob_cum_sum)
    results[steps]["acc"] = float(accuracy_cum_sum)

if not os.path.isdir(args.ckpt_folder):
    ckpt_folder = os.path.dirname(args.ckpt_folder)
else:
    ckpt_folder = args.ckpt_folder

filename = "ua" + ("" if args.ema else "_noema") + ".json"
with open(os.path.join(ckpt_folder, filename), "w") as f:
    json.dump(results, f, indent=2)
