import torch


def noise_estimation_loss(
        model,
        x0: torch.Tensor,
        t: torch.LongTensor,
        e: torch.Tensor,
        b: torch.Tensor,
        keepdim=False,
):
    a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(x, t.float())
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


def noise_estimation_loss_conditional(
        model,
        x0: torch.Tensor,
        t: torch.LongTensor,
        c: torch.LongTensor,
        e: torch.Tensor,
        b: torch.Tensor,
        cond_drop_prob=0.1,
        keepdim=False,
):
    a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(x, t.float(), c, cond_drop_prob=cond_drop_prob, mode="train")
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


def sid_sg_loss_conditional(
        model,
        x0: torch.Tensor,
        t: torch.LongTensor,
        c: torch.LongTensor,
        e: torch.Tensor,
        b: torch.Tensor,
        cond_scale=0.0,
        cond_drop_prob=0.0,
        cond_drop_mask=None,
        keepdim=False,
):
    mode = "train" if cond_scale == 0. else "test"

    a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(
        x, t.float(), c, cond_drop_prob=cond_drop_prob, cond_drop_mask=cond_drop_mask,
        mode=mode, cond_scale=cond_scale)
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


def sid_g_loss_conditional(
        p_model,
        sg_model,
        x0: torch.Tensor,
        t: torch.LongTensor,
        p_c: torch.LongTensor,
        sg_c: torch.LongTensor,
        e: torch.Tensor,
        b: torch.Tensor,
        alpha: float,
        cond_scale=0.0,
        cond_drop_prob=0.0,
        cond_drop_mask=None,
        keepdim: bool = False,
        label_to_forget=0,
        clf=None,
        pseudo_label_type="amax",
        use_diffinst=False,
):
    mode = "train" if cond_scale == 0. else "test"

    a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    bb = (1. - a).sqrt()
    aa = a.sqrt()
    x = x0 * aa + e * bb

    # if cond_drop_mask is None:
    #     cond_drop_mask = torch.rand((x.size(0), ), device=x.device) < cond_drop_prob

    if use_diffinst:
        with torch.no_grad():
            sg_e = sg_model(x, t.float(), sg_c, cond_drop_prob=cond_drop_prob, cond_drop_mask=cond_drop_mask, mode=mode,
                            cond_scale=cond_scale)
    else:
        sg_e = sg_model(x, t.float(), sg_c, cond_drop_prob=cond_drop_prob, cond_drop_mask=cond_drop_mask, mode=mode,
                        cond_scale=cond_scale)
    sg_x = (x - bb * sg_e) / aa
    if p_c is None:
        assert clf is not None
        logits = clf(sg_x.add(1.).div(2.).clamp(0, 1))
        logits[:, label_to_forget] = -torch.inf
        if pseudo_label_type == "amax":
            p_c = logits.argmax(dim=-1)
        elif pseudo_label_type == "prob":
            p_c = torch.distributions.Categorical(logits=logits).sample()
        else:
            raise NotImplementedError(pseudo_label_type)
    if use_diffinst:
        with torch.no_grad():
            p_e = p_model(x, t.float(), p_c, cond_drop_prob=cond_drop_prob, cond_drop_mask=cond_drop_mask, mode=mode,
                          cond_scale=cond_scale)
    else:
        p_e = p_model(x, t.float(), p_c, cond_drop_prob=cond_drop_prob, cond_drop_mask=cond_drop_mask, mode=mode,
                      cond_scale=cond_scale)
    p_x = (x - bb * p_e) / aa
    with torch.no_grad():
        omega = (p_x - x0).abs_().float().mean(dim=[1, 2, 3], keepdim=True).clamp_(min=0.00001)

    if use_diffinst and alpha == 1.0:
        omega = 1.0
        loss = ((p_x - sg_x) * (sg_x - x0)).div(omega).sum(dim=(1, 2, 3))
    else:
        loss = ((p_x - sg_x) * ((p_x - x0) - alpha * (p_x - sg_x))).div(omega).sum(dim=(1, 2, 3))
    if keepdim:
        return loss
    else:
        return loss.mean(dim=0)


loss_registry = {
    "simple": noise_estimation_loss,
}

loss_registry_conditional = {
    "simple": noise_estimation_loss_conditional,
    "sid-sg": sid_sg_loss_conditional,
    "sid-g": sid_g_loss_conditional,
}
