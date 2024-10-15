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
from io import StringIO


parser = ArgumentParser()
parser.add_argument("--image_folder", type=str, required=True)
parser.add_argument("--clf_path", type=str, required=True)
parser.add_argument("--config", type=str, required=True)
args = parser.parse_args()

device = "cuda"

model = torchvision.models.resnet34(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)
model.load_state_dict(torch.load(args.clf_path, map_location="cpu"))
model = model.eval().requires_grad_(False).to(device)

batch_size = 100
init_sigma = 2.5
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
if config.data.dataset == "CIFAR10":
    n_total = 5000
elif config.data.dataset == "STL10":
    n_total = 1300
shape = [3, img_size, img_size]
entropy_cum_sum = 0
forgotten_prob_cum_sum = 0
accuracy_cum_sum = 0

img_folder = glob.glob(args.image_folder + "/*.png")
n_total = len(img_folder)
img_folder = [img for img in img_folder if int(os.path.basename(img).split("_", maxsplit=1)[0]) == label_of_forgotten_class]
img_folder = [img_folder[i: i+batch_size] for i in range(0, len(img_folder), batch_size)]

for data in tqdm(img_folder):
    data = torch.stack([transform(Image.open(img)) for img in data], dim=0)
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

results["entropy"] = float(entropy_cum_sum)
results["prob"] = float(forgotten_prob_cum_sum)
results["acc"] = float(accuracy_cum_sum)

with open(os.path.join(args.image_folder, "ua.json"), "w") as f:
    json.dump(results, f)
