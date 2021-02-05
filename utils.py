import torch
import numpy as np


def get_parameter_count(model):
    total_num = 0
    for p in model.parameters():
        shape = p.shape
        num = 1
        for d in shape:
            num *= d
        total_num += num
    return total_num


def noise_input(x, banned_modality, prob=[0.5, 0.5], direction="forward", modality_noise=False):
    N = x[0].shape[0]
    D = len(x)
    dev = x[0].device
    if modality_noise:
        temp = torch.distributions.Binomial(probs=0.5)
        modality_mask = []
        for i in range(D):
            modality_mask.append(temp.sample((N, )).int().to(dev))
        global_mask = torch.ones(N, dtype=torch.int, device=dev)
        for i in range(D):
            global_mask *= (1 - modality_mask[i])
        bad_idx = global_mask == 1
        for i in range(D):
            modality_mask[i][bad_idx] = 1
            modality_mask[i] = modality_mask[i].unsqueeze(1).float()

    x_noised = []
    m = torch.distributions.Multinomial(probs=torch.tensor(prob))
    alpha = m.sample((N, )).argmax(dim=1)
    alpha = alpha.to(dev)
    for i in range(D):
        x_noised.append(x[i].clone())
        if banned_modality[i] == 1:
            x_noised[-1].zero_()
            continue

        d = x[i].shape[1] // 2
        if direction == "both":
            noise_mask = torch.ones(3, 2*d, device=dev)
            noise_mask[1, :d] = 0.
            noise_mask[2, d:] = 0.
        elif direction == "forward":
            noise_mask = torch.ones(2, 2*d, device=dev)
            noise_mask[1, d:] = 0.
        elif direction == "backward":
            noise_mask = torch.ones(2, 2*d, device=dev)
            noise_mask[1, :d] = 0.

        noise_mask = noise_mask[alpha]
        noise = - 2 * torch.ones_like(x_noised[-1], device=dev)
        if x[i].dim() == 4:
            noise_mask = noise_mask.unsqueeze(2).unsqueeze(3)
            if modality_noise:
                modality_mask[i] = modality_mask[i].unsqueeze(2).unsqueeze(3)

        x_noised[-1] = (noise_mask * x_noised[-1] + (1-noise_mask) * noise)
        if modality_noise:
            x_noised[-1] = modality_mask[i] * x_noised[-1]

    if modality_noise:
        del modality_mask[:]
    del noise_mask

    return x_noised


def txt_to_tensor(filename):
    file = open(filename, "r")
    lines = file.readlines()
    lines = [x.rstrip().split(" ") for x in lines]
    return torch.tensor(np.array(lines, dtype=np.float), dtype=torch.float)


def return_split(i, splits, tr, vl):
    if i < tr:
        return splits[0]
    elif i < vl:
        return splits[1]
    else:
        return splits[2]
