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


def noise_input(x, prob=0.5, bidirectional=False):
    x_noised = []
    m = torch.distributions.Binomial(probs=prob)
    alpha = m.sample((x[0].shape[0], ))
    alpha = alpha.to(x[0].device)
    r = torch.rand()
    for i in range(len(x)):
        d = x[i].shape[1] // 2
        x_noised.append(x[i].clone())
        noise = torch.rand_like(x_noised[-1], device=x[i].device) * 2 - 1
        if len(x[i].shape) == 4:
            alpha = alpha.reshape(-1, 1, 1, 1)
        else:
            alpha = alpha.reshape(-1, 1)

        if bidirectional:
            if r < 0.5:
                x_noised[-1][:, d:] = alpha * x_noised[-1][:, d:] + (1-alpha) * (noise[:, d:])
            else:
                x_noised[-1][:, :d] = alpha * x_noised[-1][:, :d] + (1-alpha) * (noise[:, :d])
        else:
            x_noised[-1][:, d:] = alpha * x_noised[-1][:, d:] + (1-alpha) * (noise[:, d:])

    return x_noised
