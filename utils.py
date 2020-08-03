import torch


def get_parameter_count(model):
    total_num = 0
    for p in model.parameters():
        shape = p.shape
        num = 1
        for d in shape:
            num *= d
        total_num += num
    return total_num


def noise_input(x, prob=0.5, bidirectional=False, modality_noise=True):
    N = x[0].shape[0]
    D = len(x)
    dev = x[0].device
    if modality_noise:
        temp = torch.distributions.Binomial(probs=0.5)
        modality_mask = []
        for i in range(D):
            modality_mask.append(temp.sample((N, )).int().to(dev))
            print(modality_mask[i])
        global_mask = torch.ones(N, dtype=torch.int, device=dev)
        for i in range(D):
            global_mask *= (1 - modality_mask[i])
        bad_idx = global_mask == 1
        for i in range(D):
            modality_mask[i][bad_idx] = 1
            modality_mask[i] = modality_mask[i].unsqueeze(1).float()

    x_noised = []
    m = torch.distributions.Binomial(probs=prob)
    alpha = m.sample((N, ))
    alpha = alpha.to(dev)
    for i in range(D):
        d = x[i].shape[1] // 2
        if bidirectional:
            noise_mask = torch.ones(3, 2*d, device=dev)
            noise_mask[1, :d] = 0.
            noise_mask[2, d:] = 0.
        else:
            noise_mask = torch.ones(2, 2*d, device=dev)
            noise_mask[1, d:] = 0.

        repeat_cnt = int(N // noise_mask.shape[0]) + 1
        noise_mask = noise_mask.repeat(repeat_cnt, 1)[:N]

        x_noised.append(x[i].clone())
        # noise = torch.rand_like(x_noised[-1], device=dev) * 2 - 1
        noise = torch.zeros_like(x_noised[-1], device=dev)
        if len(x[i].shape) == 4:
            noise_mask = noise_mask.unsqueeze(2).unsqueeze(3)
            if modality_noise:
                modality_mask[i] = modality_mask[i].unsqueeze(2).unsqueeze(3)

        x_noised[-1] = (noise_mask * x_noised[-1] + (1-noise_mask) * noise)
        if modality_noise:
            x_noised[-1] = modality_mask[i] * x_noised[-1]

    return x_noised
