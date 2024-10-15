import copy
import logging
import os
import pickle
import random
import re
import time

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.utils as tvu
import tqdm
from datasets import (
    all_but_one_class_path_dataset,
    data_transform,
    get_dataset,
    get_forget_dataset,
    inverse_data_transform,
)
from functions import create_class_labels, cycle, get_optimizer
from functions.denoising import generalized_steps_conditional
from functions.losses import loss_registry_conditional
from models.diffusion import Conditional_Model
from models.ema import EMAHelper
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from copy import deepcopy

import torch.distributed as tdist
from functools import partial


class InfiniteSampler:
    def __init__(self, size, rank=0, num_replicas=1, shuffle=True, seed=0, window_size=0.5):
        assert size > 0
        assert num_replicas > 0
        assert 0 <= rank < num_replicas
        assert 0 <= window_size <= 1
        self.size = size
        self.rank = rank
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.window_size = window_size

    def __iter__(self):
        order = np.arange(self.size)
        rnd = None
        window = 0
        if self.shuffle:
            rnd = np.random.RandomState(self.seed)
            rnd.shuffle(order)
            window = int(np.rint(order.size * self.window_size))

        idx = 0
        while True:
            i = idx % order.size
            if idx % self.num_replicas == self.rank:
                yield order[i]
            if window >= 2:
                j = (i - rnd.randint(window)) % order.size
                order[i], order[j] = order[j], order[i]
            idx += 1


class FixedEMAHelper(object):
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if hasattr(module, "module"):
            module = module.module
        for name, param in module.named_parameters():
            self.shadow[name] = param.data.detach().clone()

    def update(self, module):
        if hasattr(module, "module"):
            module = module.module
        for name, param in module.named_parameters():
            self.shadow[name].copy_(
                param.data.detach().lerp(self.shadow[name], self.mu))

    def ema(self, module):
        if hasattr(module, "module"):
            module = module.module
        for name, param in module.named_parameters():
            param.data.copy_(self.shadow[name])

    def ema_copy(self, module):
        if hasattr(module, "module"):
            module = module.module
        module_copy = deepcopy(module)
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict

    def to(self, device):
        for name in self.shadow:
            self.shadow[name] = self.shadow[name].to(device)


def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start**0.5,
                beta_end**0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config):
        self.args = args
        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]
        self.t_init = self.num_timesteps - 1
        if hasattr(config, "distill"):
            self.t_init = config.distill.t_init

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(self.device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def save_fim(self):
        args, config = self.args, self.config
        bs = (
            torch.cuda.device_count()
        )  # process 1 sample per GPU, so bs == number of gpus
        fim_dataset = ImageFolder(
            os.path.join(args.ckpt_folder, "class_samples"),
            transform=transforms.ToTensor(),
        )
        fim_loader = DataLoader(
            fim_dataset,
            batch_size=bs,
            num_workers=config.data.num_workers,
            shuffle=True,
        )

        print("Loading checkpoints {}".format(args.ckpt_folder))
        model = Conditional_Model(self.config)
        states = torch.load(
            os.path.join(self.args.ckpt_folder, "ckpts/ckpt.pth"),
            map_location=self.device,
        )
        model = model.to(self.device)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(states[0], strict=True)
        model.eval()

        # calculate FIM
        fisher_dict = {}
        fisher_dict_temp_list = [{} for _ in range(bs)]

        for name, param in model.named_parameters():
            fisher_dict[name] = param.data.clone().zero_()

            for i in range(bs):
                fisher_dict_temp_list[i][name] = param.data.clone().zero_()

        # calculate Fisher information diagonals
        for step, data in enumerate(
            tqdm.tqdm(fim_loader, desc="Calculating Fisher information matrix")
        ):
            x, c = data
            x, c = x.to(self.device), c.to(self.device)

            b = self.betas
            ts = torch.chunk(torch.arange(0, self.num_timesteps), args.n_chunks)

            for _t in ts:
                for i in range(len(_t)):
                    e = torch.randn_like(x)
                    t = torch.tensor([_t[i]]).expand(bs).to(self.device)

                    # keepdim=True will return loss of shape [bs], so gradients across batch are NOT averaged yet
                    if i == 0:
                        loss = loss_registry_conditional[config.model.type](
                            model, x, t, c, e, b, keepdim=True
                        )
                    else:
                        loss += loss_registry_conditional[config.model.type](
                            model, x, t, c, e, b, keepdim=True
                        )

                # store first-order gradients for each sample separately in temp dictionary
                # for each timestep chunk
                for i in range(bs):
                    model.zero_grad()
                    if i != len(loss) - 1:
                        loss[i].backward(retain_graph=True)
                    else:
                        loss[i].backward()
                    for name, param in model.named_parameters():
                        fisher_dict_temp_list[i][name] += param.grad.data
                del loss

            # after looping through all 1000 time steps, we can now aggregrate each individual sample's gradient and square and average
            for name, param in model.named_parameters():
                for i in range(bs):
                    fisher_dict[name].data += (
                        fisher_dict_temp_list[i][name].data ** 2
                    ) / len(fim_loader.dataset)
                    fisher_dict_temp_list[i][name] = (
                        fisher_dict_temp_list[i][name].clone().zero_()
                    )

            if (step + 1) % config.training.save_freq == 0:
                with open(os.path.join(args.ckpt_folder, "fisher_dict.pkl"), "wb") as f:
                    pickle.dump(fisher_dict, f)

        # save at the end
        with open(os.path.join(args.ckpt_folder, "fisher_dict.pkl"), "wb") as f:
            pickle.dump(fisher_dict, f)

    def train(self):
        args, config = self.args, self.config

        tdist.init_process_group("nccl", init_method="env://")
        rank = tdist.get_rank()
        torch.cuda.set_device(rank)
        torch.cuda.manual_seed(args.seed + rank)
        world_size = tdist.get_world_size()

        sampler_fn = partial(InfiniteSampler, rank=rank, num_replicas=world_size, seed=args.seed)

        if rank != 0:
            tdist.barrier()

        D_train_loader = get_dataset(args, config, sampler_fn=sampler_fn)
        D_train_iter = iter(D_train_loader)

        if rank == 0:
            tdist.barrier()

        model = Conditional_Model(config).train().requires_grad_(True).to(self.device)
        optimizer = get_optimizer(self.config, model.parameters())
        step_low = 0

        init_ckpt = args.init_ckpt
        if init_ckpt is not None:
            if rank != 0:
                tdist.barrier()
            init_ckpt = torch.load(init_ckpt, map_location=self.device)
            if rank == 0:
                print(f"Loading initial checkpoint from {init_ckpt}...")
            for k in list(init_ckpt[0].keys()):
                if not k.startswith("module."):
                    continue
                init_ckpt[0][k.replace("module.", "")] = init_ckpt[0].pop(k)

            model.load_state_dict(init_ckpt[0])
            optimizer.load_state_dict(init_ckpt[1])
            step_low = init_ckpt[2] + 1
            if rank == 0:
                print(f"Resuming from Step-{step_low} and finetuning for {self.config.training.n_iters} steps.")
                tdist.barrier()

        ddp_model = torch.nn.parallel.distributed.DistributedDataParallel(model)
        ddp_model.train().requires_grad_(True).to(self.device)

        if self.config.model.ema:
            ema_helper = FixedEMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
            if init_ckpt is not None and args.reuse_ema:
                if rank == 0:
                    print("Reusing the same EMA state from initial checkpoint...")
                ema_helper.load_state_dict(init_ckpt[-1])
        else:
            ema_helper = None

        cond_drop_prob = config.model.cond_drop_prob

        start = time.time()
        for step in range(step_low, self.config.training.n_iters):

            ddp_model.train()
            optimizer.zero_grad(set_to_none=True)

            x, c = next(D_train_iter)
            n = x.size(0)
            x = x.to(self.device)
            c = c.to(self.device)
            x = data_transform(self.config, x)
            e = torch.randn_like(x)
            b = self.betas.to(self.device)

            # antithetic sampling
            t = torch.randint(
                low=0, high=self.num_timesteps, size=(n // 2 + 1,)
            ).to(self.device)
            t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
            loss = loss_registry_conditional[config.model.type](
                ddp_model, x, t, c, e, b, cond_drop_prob=cond_drop_prob)

            loss.backward()
            tdist.all_reduce(loss)
            loss = loss.div_(world_size).item()

            if (step + 1) % self.config.training.log_freq == 0:
                end = time.time()
                if rank == 0:
                    logging.info(
                        f"step: {step}, loss: {loss}, time: {end-start}"
                    )
                start = time.time()

            try:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.optim.grad_clip
                )
            except Exception:
                pass
            optimizer.step()

            if self.config.model.ema:
                ema_helper.update(model)

            if (step + 1) % self.config.training.snapshot_freq == 0:
                states = [
                    model.state_dict(),
                    optimizer.state_dict(),
                    step,
                ]
                if self.config.model.ema:
                    states.append(ema_helper.state_dict())

                torch.save(
                    states,
                    os.path.join(self.config.ckpt_dir, "ckpt.pth"),
                )

                test_model = ema_helper.ema_copy(model) if self.config.model.ema else copy.deepcopy(model)
                test_model.requires_grad_(False).eval()
                self.sample_visualization(test_model, step, args.cond_scale)
                del test_model

    def train_sfd(self):
        args, config = self.args, self.config

        tdist.init_process_group("nccl", init_method="env://")
        rank = tdist.get_rank()
        torch.cuda.set_device(rank)
        torch.cuda.manual_seed(args.seed + rank)
        world_size = tdist.get_world_size()

        sampler_fn = partial(InfiniteSampler, rank=rank, num_replicas=world_size, seed=args.seed)

        if rank != 0:
            tdist.barrier()

        D_train_loader, _ = get_forget_dataset(args, config, args.label_to_forget, sampler_fn=sampler_fn)
        D_train_iter = iter(D_train_loader)

        if rank == 0:
            tdist.barrier()

        p_model = Conditional_Model(config).eval().requires_grad_(False)
        sg_model = Conditional_Model(config).eval().requires_grad_(False)
        g_model = Conditional_Model(config).eval().requires_grad_(False)

        if rank != 0:
            tdist.barrier()

        init_ckpt = torch.load(args.init_ckpt, map_location="cpu")
        if args.init_ckpt_0 is not None:
            if rank == 0:
                print(
                    "Extra checkpoint `init_ckpt_0` found! Use the score network from `init_ckpt_0`"
                    " to initialize the fake score network in `init_ckpt`."
                )
            init_ckpt_0 = torch.load(args.init_ckpt, map_location="cpu")
            init_ckpt[0] = init_ckpt_0[0]
            del init_ckpt_0
        if len(init_ckpt) <= 4:
            if rank == 0:
                print("Initialized with a pretrained DDPM checkpoint.")
            sg_model_state_dict = g_model_state_dict = init_ckpt[0]
            model_state_dicts = [init_ckpt[0]]
        else:
            if rank == 0:
                print("Initialized with a distilled DDPM checkpoint.")
            sg_model_state_dict = init_ckpt[0]
            g_model_state_dict = init_ckpt[2]
            model_state_dicts = [init_ckpt[0], init_ckpt[2]]
        ema_state_dict = init_ckpt[-1]

        try:
            init_from_ema = config.distill.init_from_ema
        except AttributeError:
            print("Falling back to default: init_from_ema=False")
            init_from_ema = False

        if init_from_ema:
            if len(init_ckpt) <= 4:
                sg_model_state_dict = g_model_state_dict = init_ckpt[-1]
                model_state_dicts = [init_ckpt[-1]]
            else:
                sg_model_state_dict = init_ckpt[0]
                g_model_state_dict = init_ckpt[-1]
                model_state_dicts = [init_ckpt[0], init_ckpt[-1]]
            ema_state_dict = None

        for model_state_dict in model_state_dicts:
            for k in list(model_state_dict.keys()):
                if "module." in k:
                    model_state_dict[k.replace("module.", "")] = model_state_dict.pop(k)
        del model_state_dicts

        if rank == 0:
            print(f"Loading model state dict from initial checkpoint at {os.path.abspath(args.init_ckpt)}")
            tdist.barrier()

        p_model.load_state_dict(sg_model_state_dict, strict=True)
        sg_model.load_state_dict(sg_model_state_dict, strict=True)
        g_model.load_state_dict(g_model_state_dict, strict=True)
        if self.config.model.ema:
            ema_helper = FixedEMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(g_model)
            if rank == 0:
                print("Loading EMA state dict from initial checkpoint...")
            if ema_state_dict is not None:
                try:
                    ema_helper.load_state_dict(ema_state_dict)
                except (TypeError, RuntimeError):
                    print("Loading EMA state dict failed! Aborting...")
            try:
                ema_helper.to(self.device)
            except AttributeError:
                pass
        else:
            ema_helper = None
        del model_state_dict, ema_state_dict, init_ckpt
        p_model.eval().requires_grad_(False).to(self.device)
        sg_model.train().requires_grad_(True).to(self.device)
        g_model.train().requires_grad_(True).to(self.device)
        if tdist.is_initialized():
            tdist.barrier()

        sg_optimizer = torch.optim.Adam(
            sg_model.parameters(),
            lr=config.sg_optim.lr,
            weight_decay=config.sg_optim.weight_decay,
            betas=(config.sg_optim.beta1, config.sg_optim.beta2),
            amsgrad=config.sg_optim.amsgrad,
            eps=config.sg_optim.eps,
        )
        g_optimizer = torch.optim.Adam(
            g_model.parameters(),
            lr=config.g_optim.lr,
            weight_decay=config.g_optim.weight_decay,
            betas=(config.g_optim.beta1, config.g_optim.beta2),
            amsgrad=config.g_optim.amsgrad,
            eps=config.g_optim.eps,
        )
        sg_model_ddp = torch.nn.parallel.distributed.DistributedDataParallel(sg_model)
        g_model_ddp = torch.nn.parallel.distributed.DistributedDataParallel(g_model)

        start = time.time()

        # distilled forgetting
        alpha = self.config.distill.alpha
        t_init = self.config.distill.t_init
        t_min = self.config.distill.t_min
        t_max = self.config.distill.t_max
        init_a = (1 - self.betas).cumprod(dim=0).sqrt()[t_init].to(self.device)
        init_b = (1 - (1 - self.betas).cumprod(dim=0)).sqrt()[t_init].to(self.device)
        init_sigma = (1 / (1 - self.betas).cumprod(dim=0) - 1.).sqrt()[t_init].to(self.device)
        try:
            cond_scale = self.config.distill.cond_scale
        except AttributeError:
            if rank == 0:
                print("cond_scale not found in config.distill!")
                print("Falling back to default cond_scale=0.0")
            cond_scale = 0.0
        try:
            cond_drop_prob = self.config.distill.cond_drop_prob
        except AttributeError:
            if rank == 0:
                print("cond_drop_prob not found in config.distill!")
                print("Falling back to default cond_drop_prob=0.0")
            cond_drop_prob = 0.0

        def x0_from_noise(xt, et):
            return xt / init_a - init_sigma * et

        def run_generator(x, c, mode="test", cond_scale=0.0, cond_drop_prob=0.0, cond_drop_mask=None, generator=None):
            t = torch.empty(x.size(0), device=self.device).fill_(t_init)
            xt = init_b * torch.randn_like(x, device=self.device)
            if generator is None:
                et = g_model(xt, t, c, mode=mode, cond_scale=cond_scale, cond_drop_prob=cond_drop_prob, cond_drop_mask=cond_drop_mask)
            else:
                et = generator(xt, t, c, mode=mode, cond_scale=cond_scale, cond_drop_prob=cond_drop_prob, cond_drop_mask=cond_drop_mask)
            x0 = x0_from_noise(xt, et)
            return x0

        config = self.config
        def sample_visualization(name, model=None):
            import math

            total_n_samples = config.training.visualization_samples
            assert total_n_samples % config.data.n_classes == 0
            n_rounds = (
                total_n_samples // config.sampling.batch_size
                if config.sampling.batch_size < total_n_samples
                else 1
            )

            c = torch.repeat_interleave(
                torch.arange(config.data.n_classes),
                total_n_samples // config.data.n_classes,
            ).to(self.device)
            c_chunks = torch.chunk(c, n_rounds, dim=0)
            img_shape = (config.data.channels, config.data.image_size, config.data.image_size,)

            rng_state = torch.cuda.get_rng_state()
            torch.cuda.manual_seed(args.seed + rank)

            with torch.no_grad():
                all_imgs = []
                for i in tqdm.tqdm(
                        range(n_rounds), desc="Generating image samples for visualization."
                ):
                    c = c_chunks[i].tensor_split(world_size, dim=0)[rank]
                    n = c.size(0)
                    m = math.ceil(len(c_chunks[i]) / world_size)
                    x = torch.randn(n, *img_shape, device=self.device)
                    x = run_generator(x, c, mode="test", cond_scale=0.0, cond_drop_prob=0.0, generator=model)
                    x = inverse_data_transform(config, x)
                    if n < m:
                        x = torch.cat([x, torch.zeros(1, *img_shape, device=self.device)], dim=0)
                    x_list = [torch.zeros(m, *img_shape, device=self.device) for _ in range(world_size)]
                    tdist.all_gather(x_list, x)
                    x = torch.cat(x_list, dim=0)
                    remainder = len(c_chunks[i]) % world_size
                    if remainder != 0:
                        inds = torch.cat([torch.arange(len(c_chunks[i]) // world_size + (j < remainder)) + \
                                          j * m for j in range(world_size)], dim=0)
                        x = x[inds]
                    all_imgs.append(x)

                all_imgs = torch.cat(all_imgs)
                grid = tvu.make_grid(
                    all_imgs,
                    nrow=total_n_samples // config.data.n_classes,
                    normalize=True,
                    padding=0,
                )

                if rank == 0:
                    try:
                        tvu.save_image(
                            grid, os.path.join(self.config.log_dir, f"sample-{name}.png")
                        )  # if called during training of base model
                    except AttributeError:
                        tvu.save_image(
                            grid, os.path.join(self.args.ckpt_folder, f"sample-{name}.png")
                        )  # if called from sample.py
                tdist.barrier()

            torch.cuda.set_rng_state(rng_state)

        sg_remain_coef = self.config.distill.sg_remain_coef
        sg_forget_coef = self.config.distill.sg_forget_coef
        g_remain_coef = self.config.distill.g_remain_coef
        g_forget_coef = self.config.distill.g_forget_coef
        forget_warmup = getattr(self.config.distill, "forget_warmup", 0)
        forget_warmup = int(forget_warmup * self.config.training.n_iters + 0.5)
        label_to_forget = args.label_to_forget
        label_to_override = self.config.distill.label_to_override
        pseudo_label_type = self.config.distill.pseudo_label_type
        use_clf = self.config.distill.use_clf
        clf_path = self.config.distill.clf_path
        use_diffinst = getattr(self.config.distill, "use_diffinst", False)
        if use_clf and clf_path is not None:
            clf = torch.jit.load(clf_path).to(self.device)
        else:
            clf = None
        for step in range(0, self.config.training.n_iters):
            # sg step
            # remain_loss
            sg_model_ddp.train().requires_grad_(True)
            sg_optimizer.zero_grad(set_to_none=True)
            x, c = next(D_train_iter)
            c = c.to(self.device)
            n = x.size(0)
            cond_drop_mask = torch.rand(n, device=self.device) < cond_drop_prob
            mode = "test" if cond_drop_prob == 0.0 else "train"
            with torch.no_grad():
                x0 = run_generator(x, c, mode=mode, cond_scale=0.0, cond_drop_mask=cond_drop_mask)

            x0 = x0.to(self.device)  # [-1, 1]
            e = torch.randn_like(x0)
            b = self.betas.to(self.device)

            t = torch.randint(low=t_min, high=t_max, size=(n,)).to(self.device)
            sg_remain_loss = loss_registry_conditional[config.model.type + "-sg"](
                sg_model_ddp, x0, t, c, e, b, cond_scale=cond_scale, cond_drop_prob=cond_drop_prob, cond_drop_mask=cond_drop_mask)
            sg_remain_loss.mul(sg_remain_coef).backward()
            sg_remain_loss = sg_remain_loss.detach()
            tdist.all_reduce(sg_remain_loss)
            sg_remain_loss = sg_remain_loss.div_(world_size).item()

            # forget loss
            if sg_forget_coef > 0:
                forget_c = torch.empty_like(c).fill_(label_to_forget)
                with torch.no_grad():
                    x0 = run_generator(x, forget_c, mode="test", cond_scale=0.0, cond_drop_prob=0.0, cond_drop_mask=None)

                x0 = x0.to(self.device)
                e = torch.randn_like(x0)
                if step < forget_warmup:
                    t = torch.randint(
                        low=int(0.02 * self.num_timesteps + 0.5),
                        high=self.num_timesteps - int(0.02 * self.num_timesteps + 0.5),
                        size=(n,)).to(self.device)
                else:
                    t = torch.randint(low=t_min, high=t_max, size=(n,)).to(self.device)
                sg_forget_loss = loss_registry_conditional[config.model.type + "-sg"](
                    sg_model_ddp, x0, t, forget_c, e, b, cond_scale=cond_scale, cond_drop_prob=0.0, cond_drop_mask=None)
                sg_forget_loss.mul(sg_forget_coef).backward()
                sg_forget_loss = sg_forget_loss.detach()
                tdist.all_reduce(sg_forget_loss)
                sg_forget_loss = sg_forget_loss.div_(world_size).item()
            else:
                sg_forget_loss = 0

            try:
                torch.nn.utils.clip_grad_value_(sg_model.parameters(), config.sg_optim.grad_clip)
            except Exception as e:
                print(e)
            sg_optimizer.step()

            # g step
            # remain_loss
            sg_model.eval().requires_grad_(False)
            g_model_ddp.train().requires_grad_(True)
            g_optimizer.zero_grad(set_to_none=True)
            cond_drop_mask = torch.rand(n, device=self.device) < cond_drop_prob
            mode = "test" if cond_drop_prob == 0.0 else "train"
            x0 = run_generator(x, c, mode=mode, cond_scale=0.0, cond_drop_prob=0.0, cond_drop_mask=cond_drop_mask, generator=g_model_ddp)
            x0 = x0.to(self.device)
            e = torch.randn_like(x0)
            t = torch.randint(low=t_min, high=t_max, size=(n,)).to(self.device)
            g_remain_loss = loss_registry_conditional[config.model.type + "-g"](
                p_model, sg_model, x0, t, c, c, e, b, alpha=alpha,
                cond_scale=cond_scale, cond_drop_prob=0.0, cond_drop_mask=cond_drop_mask,
                use_diffinst=use_diffinst,
            )
            g_remain_loss.mul(g_remain_coef).backward()
            g_remain_loss = g_remain_loss.detach()
            tdist.all_reduce(g_remain_loss)
            g_remain_loss = g_remain_loss.div_(world_size).item()

            # forget loss
            if g_forget_coef > 0:
                forget_c = torch.empty_like(c).fill_(label_to_forget)
                if label_to_override is not None:
                    override_c = torch.empty_like(c, device=self.device).fill_(label_to_override)
                else:
                    if pseudo_label_type == "rand":
                        override_c = torch.randint_like(c, low=0, high=9)
                        override_c[override_c == label_to_forget] = 9
                    else:
                        override_c = None

                x0 = run_generator(x, forget_c, mode="train", cond_scale=0.0, cond_drop_prob=0.0, cond_drop_mask=None, generator=g_model_ddp)
                x0 = x0.to(self.device)
                e = torch.randn_like(x0)

                if step < forget_warmup:
                    t = torch.randint(
                        low=int(0.02 * self.num_timesteps + 0.5),
                        high=self.num_timesteps - int(0.02 * self.num_timesteps + 0.5),
                        size=(n,)).to(self.device)
                else:
                    t = torch.randint(low=t_min, high=t_max, size=(n,)).to(self.device)
                g_forget_loss = loss_registry_conditional[config.model.type + "-g"](
                    p_model, sg_model, x0, t, p_c=override_c, sg_c=forget_c, e=e, b=b,
                    label_to_forget=label_to_forget, alpha=alpha, clf=clf, pseudo_label_type=pseudo_label_type,
                    cond_scale=cond_scale, cond_drop_prob=0.0, cond_drop_mask=None,
                    use_diffinst=use_diffinst,
                )
                g_forget_loss.mul(g_forget_coef).backward()
                g_forget_loss = g_forget_loss.detach()
                tdist.all_reduce(g_forget_loss)
                g_forget_loss = g_forget_loss.div_(world_size).item()
            else:
                g_forget_loss = 0

            try:
                torch.nn.utils.clip_grad_value_(g_model.parameters(), config.g_optim.grad_clip)
            except Exception as e:
                print(e)
            g_optimizer.step()
            g_model.eval().requires_grad_(False)

            if (step + 1) % self.config.training.log_freq == 0:
                end = time.time()
                if rank == 0:
                    logging.info(
                        f"step: {step}\n"
                        f"time: {end - start}\n"
                        f"sg_remain_loss: {sg_remain_loss}\n"
                        f"sg_forget_loss: {sg_forget_loss}\n"
                        f"sg_total_loss: {sg_remain_loss * sg_remain_coef + sg_forget_loss * sg_forget_coef}\n"
                        f"g_remain_loss: {g_remain_loss}\n"
                        f"g_forget_loss: {g_forget_loss}\n"
                        f"g_total_loss: {g_remain_loss * g_remain_coef + g_forget_loss * g_forget_coef}\n\n"
                    )
                start = time.time()

            if self.config.model.ema:
                ema_helper.update(g_model)

            early_stage_n_intervals = getattr(self.config.training, "early_stage_n_intervals", None)
            if early_stage_n_intervals is not None:
                early_stage = np.exp(np.linspace(
                    np.log(self.config.training.snapshot_freq) // early_stage_n_intervals,
                    np.log(self.config.training.snapshot_freq), early_stage_n_intervals)).astype(np.int64)[:-1]
            else:
                early_stage = []

            if ((step + 1) % self.config.training.snapshot_freq == 0) or ((step + 1) in early_stage):
                states = [
                    sg_model.state_dict(),
                    sg_optimizer.state_dict(),
                    g_model.state_dict(),
                    g_optimizer.state_dict(),
                    step,
                ]
                if self.config.model.ema:
                    states.append(ema_helper.state_dict())

                if (step + 1) in early_stage:
                        states[0] = states[1] = states[3] = None

                if rank == 0:
                    torch.save(
                        states,
                        os.path.join(self.config.ckpt_dir, f"ckpt-{step}.pth"),
                    )
                if tdist.is_initialized():
                    tdist.barrier()

                test_model = ema_helper.ema_copy(g_model) if self.config.model.ema else copy.deepcopy(g_model)
                test_model.eval().requires_grad_(False).to(self.device)
                name = f"ema-{step}" if self.config.model.ema else step
                sample_visualization(name, model=test_model)
                del test_model
                if self.config.model.ema:
                    sample_visualization(step)


    def train_forget(self):
        args, config = self.args, self.config
        logging.info(
            f"Training diffusion forget with contrastive and EWC. Gamma: {config.training.gamma}, lambda: {config.training.lmbda}"
        )
        D_train_loader = all_but_one_class_path_dataset(
            config,
            os.path.join(args.ckpt_folder, "class_samples"),
            args.label_to_forget,
        )
        D_train_iter = cycle(D_train_loader)

        print("Loading checkpoints {}".format(args.ckpt_folder))
        model = Conditional_Model(config)
        states = torch.load(
            os.path.join(args.ckpt_folder, "ckpts/ckpt.pth"),
            map_location=self.device,
        )
        model = model.to(self.device)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(states[0], strict=True)
        optimizer = get_optimizer(config, model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=config.model.ema_rate)
            ema_helper.register(model)
            ema_helper.load_state_dict(states[-1])
        else:
            ema_helper = None

        with open(os.path.join(args.ckpt_folder, "fisher_dict.pkl"), "rb") as f:
            fisher_dict = pickle.load(f)

        params_mle_dict = {}
        for name, param in model.named_parameters():
            params_mle_dict[name] = param.data.clone()

        label_choices = list(range(config.data.n_classes))
        label_choices.remove(args.label_to_forget)

        for step in range(0, config.training.n_iters):
            model.train()
            x_remember, c_remember = next(D_train_iter)
            x_remember, c_remember = x_remember.to(self.device), c_remember.to(
                self.device
            )
            x_remember = data_transform(config, x_remember)

            n = x_remember.size(0)
            channels = config.data.channels
            img_size = config.data.image_size
            c_forget = (torch.ones(n, dtype=int) * args.label_to_forget).to(self.device)
            x_forget = (
                torch.rand((n, channels, img_size, img_size), device=self.device) - 0.5
            ) * 2.0
            e_remember = torch.randn_like(x_remember)
            e_forget = torch.randn_like(x_forget)
            b = self.betas

            # antithetic sampling
            t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(
                self.device
            )
            t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
            loss = loss_registry_conditional[config.model.type](
                model, x_forget, t, c_forget, e_forget, b, cond_drop_prob=0.0
            ) + config.training.gamma * loss_registry_conditional[config.model.type](
                model, x_remember, t, c_remember, e_remember, b, cond_drop_prob=0.0
            )
            forgetting_loss = loss.item()

            ewc_loss = 0.0
            for name, param in model.named_parameters():
                _loss = (
                    fisher_dict[name].to(self.device)
                    * (param - params_mle_dict[name].to(self.device)) ** 2
                )
                loss += config.training.lmbda * _loss.sum()
                ewc_loss += config.training.lmbda * _loss.sum()

            if (step + 1) % config.training.log_freq == 0:
                logging.info(
                    f"step: {step}, loss: {loss.item()}, forgetting loss: {forgetting_loss}, ewc loss: {ewc_loss}"
                )

            optimizer.zero_grad()
            loss.backward()

            try:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.optim.grad_clip
                )
            except Exception:
                pass

            optimizer.step()

            if self.config.model.ema:
                ema_helper.update(model)

            if (step + 1) % config.training.snapshot_freq == 0:
                states = [
                    model.state_dict(),
                    optimizer.state_dict(),
                    # epoch,
                    step,
                ]
                if config.model.ema:
                    states.append(ema_helper.state_dict())

                torch.save(
                    states,
                    os.path.join(config.ckpt_dir, "ckpt.pth"),
                )

                test_model = (
                    ema_helper.ema_copy(model)
                    if config.model.ema
                    else copy.deepcopy(model)
                )
                test_model.eval()
                self.sample_visualization(test_model, step, args.cond_scale)
                del test_model

    def retrain(self):
        args, config = self.args, self.config

        D_remain_loader, _ = get_forget_dataset(
            args, config, args.label_to_forget
        )
        D_remain_iter = cycle(D_remain_loader)

        model = Conditional_Model(config)

        optimizer = get_optimizer(self.config, model.parameters())
        model.to(self.device)
        model = torch.nn.DataParallel(model)

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        model.train()

        start = time.time()
        for step in range(0, self.config.training.n_iters):
            model.train()
            x, c = next(D_remain_iter)

            n = x.size(0)
            x = x.to(self.device)
            x = data_transform(self.config, x)
            e = torch.randn_like(x)
            b = self.betas

            # antithetic sampling
            t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(
                self.device
            )
            t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
            loss = loss_registry_conditional[config.model.type](model, x, t, c, e, b)

            if (step + 1) % self.config.training.log_freq == 0:
                end = time.time()
                logging.info(f"step: {step}, loss: {loss.item()}, time: {end-start}")
                start = time.time()

            optimizer.zero_grad()
            loss.backward()

            try:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.optim.grad_clip
                )
            except Exception:
                pass
            optimizer.step()

            if self.config.model.ema:
                ema_helper.update(model)

            if (step + 1) % self.config.training.snapshot_freq == 0:
                states = [
                    model.state_dict(),
                    optimizer.state_dict(),
                    step,
                ]
                if self.config.model.ema:
                    states.append(ema_helper.state_dict())

                torch.save(
                    states,
                    os.path.join(self.config.ckpt_dir, f"ckpt-{step}.pth"),
                )

                test_model = (
                    ema_helper.ema_copy(model)
                    if self.config.model.ema
                    else copy.deepcopy(model)
                )
                test_model.eval()
                self.sample_visualization(test_model, step, args.cond_scale)
                del test_model

    def saliency_unlearn(self):
        args, config = self.args, self.config

        D_remain_loader, D_forget_loader = get_forget_dataset(
            args, config, args.label_to_forget
        )
        D_remain_iter = cycle(D_remain_loader)
        D_forget_iter = cycle(D_forget_loader)

        print("Loading mask {}".format(args.mask_path), end="")

        if args.mask_path:
            mask = torch.load(args.mask_path)
            print("...done!")
        else:
            mask = None

        print("Loading checkpoints {}".format(args.ckpt_folder))

        model = Conditional_Model(config)
        states = torch.load(
            os.path.join(args.ckpt_folder, "ckpts/ckpt.pth"),
            map_location=self.device,
        )
        model = model.to(self.device)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(states[0], strict=True)
        optimizer = get_optimizer(config, model.parameters())
        criteria = torch.nn.MSELoss()

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=config.model.ema_rate)
            ema_helper.register(model)
            ema_helper.load_state_dict(states[-1])
        else:
            ema_helper = None

        model.train()
        start = time.time()
        for step in range(0, self.config.training.n_iters):
            model.train()

            # remain stage
            remain_x, remain_c = next(D_remain_iter)
            n = remain_x.size(0)
            remain_x = remain_x.to(self.device)
            remain_x = data_transform(self.config, remain_x)
            e = torch.randn_like(remain_x)
            b = self.betas

            t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(
                self.device
            )
            t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
            remain_loss = loss_registry_conditional[config.model.type](
                model, remain_x, t, remain_c, e, b
            )

            # forget stage
            forget_x, forget_c = next(D_forget_iter)

            n = forget_x.size(0)
            forget_x = forget_x.to(self.device)
            forget_x = data_transform(self.config, forget_x)
            e = torch.randn_like(forget_x)
            b = self.betas

            t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(
                self.device
            )
            t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]

            if args.method == "ga":
                forget_loss = -loss_registry_conditional[config.model.type](
                    model, forget_x, t, forget_c, e, b
                )

            else:
                a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
                forget_x = forget_x * a.sqrt() + e * (1.0 - a).sqrt()

                output = model(forget_x, t.float(), forget_c, mode="train")

                if args.method == "rl":
                    pseudo_c = torch.full(
                        forget_c.shape,
                        (args.label_to_forget + 1) % 10,
                        device=forget_c.device,
                    )
                    pseudo = model(forget_x, t.float(), pseudo_c, mode="train").detach()
                    forget_loss = criteria(pseudo, output)

            loss = forget_loss + args.alpha * remain_loss

            l1_loss = None
            if args.l1_sparse:
                l1_loss = 0
                for param in model.parameters():
                    if param.requires_grad:
                        l1_loss += (param - 1).abs().sum()
                loss += args.l1_coef * l1_loss

            if (step + 1) % self.config.training.log_freq == 0:
                end = time.time()
                logging.info(
                    f"step: {step}, " 
                    f"forget loss: {forget_loss.item()}, "
                    f"remain loss: {remain_loss.item()}, "
                    f"{'' if l1_loss is None else ('l1 loss:' + str(l1_loss.item())) + ', '}"
                    f"total loss: {loss.item()}, "
                    f"time: {end-start}"
                )
                start = time.time()

            optimizer.zero_grad()
            loss.backward()

            try:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.optim.grad_clip
                )
            except Exception:
                pass

            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name].to(param.grad.device)
            optimizer.step()

            if self.config.model.ema:
                ema_helper.update(model)

            if (step + 1) % self.config.training.snapshot_freq == 0:
                states = [
                    model.state_dict(),
                    optimizer.state_dict(),
                    step,
                ]
                if self.config.model.ema:
                    states.append(ema_helper.state_dict())

                torch.save(
                    states,
                    os.path.join(self.config.ckpt_dir, f"ckpt-{step}.pth"),
                )

                test_model = (
                    ema_helper.ema_copy(model)
                    if self.config.model.ema
                    else copy.deepcopy(model)
                )
                test_model.eval()
                self.sample_visualization(test_model, step, args.cond_scale)
                del test_model


    def train_esd(self):
        args, config = self.args, self.config

        print("Loading checkpoints {}".format(args.ckpt_folder))

        states = torch.load(
            os.path.join(args.ckpt_folder, "ckpts/ckpt.pth"),
            map_location=self.device,
        )

        new_states = {}
        for key, value in states[0].items():
            new_states[key.replace("module.", "")] = value

        model = Conditional_Model(config).to(self.device)
        model.load_state_dict(new_states, strict=True)

        model_orig = Conditional_Model(config).to(self.device)
        model_orig.load_state_dict(new_states, strict=True)

        criteria = torch.nn.MSELoss()
        optimizer = get_optimizer(self.config, model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        ddim_steps = 50
        c = torch.tensor([args.label_to_forget]).to(self.device)

        for step in tqdm.tqdm(range(0, 10000)):
            model.train()
            t_enc = random.randint(0, 50)

            og_num = round((int(t_enc) / ddim_steps) * 1000)
            og_num_lim = round((int(t_enc + 1) / ddim_steps) * 1000)
            t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=self.device)

            size = config.data.image_size
            e = torch.randn((1, 3, size, size)).to(self.device)

            with torch.no_grad():
                # https://github.com/clear-nus/selective-amnesia/blob/a7a27ab573ba3be77af9e7aae4a3095da9b136ac/ddpm/runners/diffusion.py#L451

                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, t_enc, skip)
                x = generalized_steps_conditional(
                    e, c, seq, model, self.betas, args.cond_scale, eta=self.args.eta
                )[0][int(t_enc)].to(self.device)

                e_0 = model_orig.forward(
                    x, t_enc_ddpm, c, cond_drop_prob=1.0, mode="train"
                )
                e_p = model_orig.forward(
                    x, t_enc_ddpm, c, cond_drop_prob=0.0, mode="train"
                )

            e_n = model.forward(
                x, t_enc_ddpm.float(), c, cond_drop_prob=0.0, mode="train"
            )
            e_0.requires_grad = False
            e_p.requires_grad = False

            loss = criteria(e_n, e_0 - (args.negative_guidance * (e_p - e_0)))

            try:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.optim.grad_clip
                )
            except Exception:
                pass
            optimizer.step()
            loss.backward()

            if self.config.model.ema:
                ema_helper.update(model)

            if (step + 1) % self.config.training.snapshot_freq == 0:
                states = [
                    model.state_dict(),
                    optimizer.state_dict(),
                    step,
                ]
                if self.config.model.ema:
                    states.append(ema_helper.state_dict())

                torch.save(
                    states,
                    os.path.join(self.config.ckpt_dir, f"ckpt-{step}.pth"),
                )

                test_model = (
                    ema_helper.ema_copy(model)
                    if self.config.model.ema
                    else copy.deepcopy(model)
                )
                test_model.eval()
                self.sample_visualization(test_model, step, args.cond_scale)
                del test_model

    def load_ema_model(self):
        model = Conditional_Model(self.config)
        states = torch.load(
            os.path.join(self.args.ckpt_folder, "ckpts/ckpt.pth"),
            map_location=self.device,
        )
        model = model.to(self.device)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(states[0], strict=True)

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
            ema_helper.load_state_dict(states[-1])
            test_model = ema_helper.ema_copy(model)
        else:
            ema_helper = None

        model.eval()
        return model

    def sample(self):
        model = Conditional_Model(self.config)
        if self.args.ckpt_folder.endswith(".pth"):
            ckpt_path = self.args.ckpt_folder
            self.args.ckpt_folder = os.path.dirname(
                os.path.dirname(ckpt_path))
        else:
            ckpt_path = os.path.join(self.args.ckpt_folder, "ckpts/ckpt.pth")
        if not os.path.exists(ckpt_path):
            import glob
            ckpt_path = sorted(
                glob.glob(os.path.join(self.args.ckpt_folder, "**/ckpt-*.pth"), recursive=True),
                key=lambda x: os.path.getctime(x)
            )[-1]
            print(f"Loading checkpoint from {ckpt_path}...")
        states = torch.load(ckpt_path, map_location=self.device)

        model = model.to(self.device)
        model_state = states[0] if len(states) <= 4 else states[2]

        for k in list(model_state.keys()):
            if k.startswith("module."):
                model_state[k[7:]] = model_state.pop(k)
        if hasattr(states[-1], "keys"):
            for k in list(states[-1].keys()):
                if k.startswith("module."):
                    states[-1][k[7:]] = states[-1].pop(k)

        model.load_state_dict(model_state, strict=True)
        model.requires_grad_(False).eval()
        model = torch.nn.DataParallel(model)

        if self.config.model.ema:
            print("Using EMA...")
            ema_helper = FixedEMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
            ema_helper.load_state_dict(states[-1])
            test_model = torch.nn.DataParallel(ema_helper.ema_copy(model))
        else:
            test_model = model

        del states, model_state

        if self.args.sample_type == "one_step":
            t_init = self.t_init
            init_a = (1 - self.betas).cumprod(dim=0).sqrt()[t_init]
            init_b = (1 - (1 - self.betas).cumprod(dim=0)).sqrt()[t_init]
            init_sigma = (1 / (1 - self.betas).cumprod(dim=0) - 1.).sqrt()[t_init]
            print(f"init_t: {t_init}")
            print(f"init_a: {init_a}")
            print(f"init_b: {init_b}")
            print(f"init_sigma: {init_sigma}")

        if self.args.mode == "sample_fid":
            self.sample_fid(test_model, self.args.cond_scale)
        elif self.args.mode == "sample_classes":
            self.sample_classes(test_model, self.args.cond_scale)
        elif self.args.mode == "visualization":
            self.sample_visualization(
                model, str(self.args.cond_scale), self.args.cond_scale
            )

    def sample_classes(self, model, cond_scale):
        """
        Samples each class from the model. Can be used to calculate FIM, for generative replay
        or for classifier evaluation. Stores samples in "./class_samples/<class_label>".
        """
        config = self.config
        args = self.args
        sample_dir = os.path.join(args.ckpt_folder, "class_samples")
        os.makedirs(sample_dir, exist_ok=True)
        img_id = 0
        classes, _ = create_class_labels(
            args.classes_to_generate, n_classes=config.data.n_classes
        )
        n_samples_per_class = args.n_samples_per_class

        for i in classes:
            os.makedirs(os.path.join(sample_dir, str(i)), exist_ok=True)
            if n_samples_per_class % config.sampling.batch_size == 0:
                n_rounds = n_samples_per_class // config.sampling.batch_size
            else:
                n_rounds = n_samples_per_class // config.sampling.batch_size + 1
            n_left = n_samples_per_class  # tracker on how many samples left to generate

            with torch.no_grad():
                for j in tqdm.tqdm(
                    range(n_rounds),
                    desc=f"Generating image samples for class {i} to use as dataset",
                ):
                    if n_left >= config.sampling.batch_size:
                        n = config.sampling.batch_size
                    else:
                        n = n_left

                    x = torch.randn(
                        n,
                        config.data.channels,
                        config.data.image_size,
                        config.data.image_size,
                        device=self.device,
                    )
                    c = torch.ones(x.size(0), device=self.device, dtype=int) * int(i)
                    x = self.sample_image(x, model, c, cond_scale)
                    x = inverse_data_transform(config, x)

                    for k in range(n):
                        tvu.save_image(
                            x[k],
                            os.path.join(sample_dir, str(c[k].item()), f"{img_id}.png"),
                            normalize=True,
                        )
                        img_id += 1

                    n_left -= n

    def sample_one_class(self, model, cond_scale, class_label):
        """
        Samples one class only for classifier evaluation.
        """
        config = self.config
        args = self.args
        sample_dir = os.path.join(args.ckpt_folder, "class_" + str(class_label))
        os.makedirs(sample_dir, exist_ok=True)
        img_id = 0
        total_n_samples = 500

        if total_n_samples % config.sampling.batch_size == 0:
            n_rounds = total_n_samples // config.sampling.batch_size
        else:
            n_rounds = total_n_samples // config.sampling.batch_size + 1
        n_left = total_n_samples  # tracker on how many samples left to generate

        with torch.no_grad():
            for j in tqdm.tqdm(
                range(n_rounds),
                desc=f"Generating image samples for class {class_label}",
            ):
                if n_left >= config.sampling.batch_size:
                    n = config.sampling.batch_size
                else:
                    n = n_left

                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )
                c = torch.ones(x.size(0), device=self.device, dtype=int) * class_label
                x = self.sample_image(x, model, c, cond_scale)
                x = inverse_data_transform(config, x)

                for k in range(n):
                    tvu.save_image(
                        x[k], os.path.join(sample_dir, f"{img_id}.png"), normalize=True
                    )
                    img_id += 1

                n_left -= n

    def sample_fid(self, model, cond_scale):
        config = self.config
        args = self.args
        classes, excluded_classes = create_class_labels(
            args.classes_to_generate, n_classes=config.data.n_classes
        )
        n_samples_per_class = args.n_samples_per_class

        sample_dir = f"fid_samples_guidance_{args.cond_scale}"
        sample_dir += os.environ.get("SAMPLE_DIR_SUFFIX", "")
        if excluded_classes:
            excluded_classes_str = "_".join(str(i) for i in excluded_classes)
            sample_dir = f"{sample_dir}_excluded_class_{excluded_classes_str}"
        sample_dir = os.path.join(args.ckpt_folder, sample_dir)
        os.makedirs(sample_dir, exist_ok=True)

        for i in classes:
            img_id = getattr(args.start_img_id, "start_img_id", 0)

            if n_samples_per_class % config.sampling.batch_size == 0:
                n_rounds = n_samples_per_class // config.sampling.batch_size
            else:
                n_rounds = n_samples_per_class // config.sampling.batch_size + 1
            n_left = n_samples_per_class  # tracker on how many samples left to generate

            with torch.no_grad():
                for j in tqdm.tqdm(
                    range(n_rounds),
                    desc=f"Generating image samples for class {i} for FID",
                ):
                    if n_left >= config.sampling.batch_size:
                        n = config.sampling.batch_size
                    else:
                        n = n_left

                    x = torch.randn(
                        n,
                        config.data.channels,
                        config.data.image_size,
                        config.data.image_size,
                        device=self.device,
                    )
                    c = torch.ones(x.size(0), device=self.device, dtype=int) * int(i)
                    x = self.sample_image(x, model, c, cond_scale)
                    x = inverse_data_transform(config, x)

                    for k in range(n):
                        tvu.save_image(
                            x[k],
                            os.path.join(sample_dir, f"{i:03d}_{img_id:05d}.png"),
                            normalize=False,
                        )
                        img_id += 1

                    n_left -= n

    def sample_image(self, x, model, c, cond_scale, last=True):
        self.betas = self.betas.to(self.device)

        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import generalized_steps_conditional

            xs = generalized_steps_conditional(
                x, c, seq, model, self.betas, cond_scale, eta=self.args.eta
            )
            x = xs
        elif self.args.sample_type == "ddpm_noisy":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import ddpm_steps_conditional

            x = ddpm_steps_conditional(x, c, seq, model, self.betas)
        elif self.args.sample_type == "one_step":
            t_init = self.t_init
            init_a = (1 - self.betas).cumprod(dim=0).sqrt()[t_init].to(self.device)
            init_b = (1 - (1 - self.betas).cumprod(dim=0)).sqrt()[t_init].to(self.device)
            init_sigma = (1 / (1 - self.betas).cumprod(dim=0) - 1.).sqrt()[t_init].to(self.device)
            t = torch.empty(x.size(0), device=self.device).fill_(t_init)
            # xt = torch.randn_like(x, device=self.device)
            # xt = init_sigma * torch.randn_like(x, device=self.device)
            xt = init_b * torch.randn_like(x, device=self.device)
            et = model(xt, t, c, mode="test", cond_drop_prob=0.0, cond_scale=self.args.cond_scale)
            x = xt / init_a - init_sigma * et
        else:
            raise NotImplementedError
        if last and self.args.sample_type != "one_step":
            x = x[0][-1]
        return x

    def sample_visualization(self, model, name, cond_scale):
        import math

        config = self.config
        args = self.args
        total_n_samples = config.training.visualization_samples
        assert total_n_samples % config.data.n_classes == 0
        n_rounds = (
            total_n_samples // config.sampling.batch_size
            if config.sampling.batch_size < total_n_samples
            else 1
        )

        # esd
        c = torch.repeat_interleave(
            torch.arange(config.data.n_classes),
            total_n_samples // config.data.n_classes,
        ).to(self.device)

        c_chunks = torch.chunk(c, n_rounds, dim=0)

        world_size = 1
        rank = 0
        if tdist.is_initialized():
            world_size = max(tdist.get_world_size(), 1)
            rank = max(tdist.get_rank(), 0)

        img_shape = (
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
        )

        rng_state = torch.cuda.get_rng_state()
        torch.cuda.manual_seed(args.seed + rank)
        with torch.no_grad():
            all_imgs = []
            for i in tqdm.tqdm(
                range(n_rounds), desc="Generating image samples for visualization."
            ):
                c = c_chunks[i].tensor_split(world_size, dim=0)[rank]
                n = c.size(0)
                m = math.ceil(len(c_chunks[i]) / world_size)
                x = torch.randn(n, *img_shape, device=self.device)
                x = self.sample_image(x, model, c, cond_scale).to(self.device)
                x = inverse_data_transform(config, x)
                if n < m:
                    x = torch.cat([x, torch.zeros(1, *img_shape, device=self.device)], dim=0)
                x_list = [torch.zeros(m, *img_shape, device=self.device) for _ in range(world_size)]
                tdist.all_gather(x_list, x)
                x = torch.cat(x_list, dim=0)
                remainder = len(c_chunks[i]) % world_size
                if remainder != 0:
                    inds = torch.cat([torch.arange(len(c_chunks[i]) // world_size + (j < remainder)) + \
                                      j * m for j in range(world_size)], dim=0)
                    x = x[inds]
                all_imgs.append(x)

            all_imgs = torch.cat(all_imgs).cpu()
            grid = tvu.make_grid(
                all_imgs,
                nrow=total_n_samples // config.data.n_classes,
                normalize=True,
                padding=0,
            )

            if rank == 0:
                try:
                    tvu.save_image(
                        grid, os.path.join(self.config.log_dir, f"sample-{name}.png")
                    )  # if called during training of base model
                except AttributeError:
                    tvu.save_image(
                        grid, os.path.join(self.args.ckpt_folder, f"sample-{name}.png")
                    )  # if called from sample.py
            if tdist.is_initialized():
                tdist.barrier()

        torch.cuda.set_rng_state(rng_state)

    def generate_mask(self):
        args, config = self.args, self.config
        logging.info(
            f"Generating mask of diffusion to achieve gradient sparsity. Gamma: {config.training.gamma}, lambda: {config.training.lmbda}"
        )

        dataset_name = config.data.dataset.lower()
        _, D_forget_loader = get_forget_dataset(args, config, args.label_to_forget)

        print("Loading checkpoints {}".format(args.ckpt_folder))
        model = Conditional_Model(config)
        states = torch.load(
            os.path.join(args.ckpt_folder, "ckpts/ckpt.pth"),
            map_location=self.device,
        )
        model = model.to(self.device)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(states[0], strict=True)

        optimizer = get_optimizer(config, model.parameters())

        gradients = {}
        for name, param in model.named_parameters():
            gradients[name] = 0

        model.eval()

        for x, forget_c in D_forget_loader:
            n = x.size(0)
            x = x.to(self.device)
            x = data_transform(self.config, x)
            e = torch.randn_like(x)
            b = self.betas

            # antithetic sampling
            t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(
                self.device
            )
            t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]

            # loss 1
            a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
            x = x * a.sqrt() + e * (1.0 - a).sqrt()
            output = model(
                x, t.float(), forget_c, cond_scale=args.cond_scale, mode="test"
            )

            # https://github.com/clear-nus/selective-amnesia/blob/a7a27ab573ba3be77af9e7aae4a3095da9b136ac/ddpm/models/diffusion.py#L338
            loss = (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)

            optimizer.zero_grad()
            loss.backward()

            try:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.optim.grad_clip
                )
            except Exception:
                pass

            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        gradient = param.grad.data.cpu()
                        gradients[name] += gradient

        with torch.no_grad():

            for name in gradients:
                gradients[name] = torch.abs_(gradients[name])

            mask_path = os.path.join(f'results/{dataset_name}/mask', str(args.label_to_forget))
            os.makedirs(mask_path, exist_ok=True)

            threshold_list = [args.mask_ratio]
            for i in threshold_list:
                print(i)
                sorted_dict_positions = {}
                hard_dict = {}

                # Concatenate all tensors into a single tensor
                all_elements = - torch.cat(
                    [tensor.flatten() for tensor in gradients.values()]
                )

                # Calculate the threshold index for the top 10% elements
                threshold_index = int(len(all_elements) * i)

                # Calculate positions of all elements
                positions = torch.argsort(all_elements)
                ranks = torch.argsort(positions)

                start_index = 0
                for key, tensor in gradients.items():
                    num_elements = tensor.numel()
                    tensor_ranks = ranks[start_index : start_index + num_elements]

                    sorted_positions = tensor_ranks.reshape(tensor.shape)
                    sorted_dict_positions[key] = sorted_positions

                    # Set the corresponding elements to 1
                    threshold_tensor = torch.zeros_like(tensor_ranks)
                    threshold_tensor[tensor_ranks < threshold_index] = 1
                    threshold_tensor = threshold_tensor.reshape(tensor.shape)
                    hard_dict[key] = threshold_tensor
                    start_index += num_elements

                torch.save(hard_dict, os.path.join(mask_path, f'with_{str(i)}.pt'))