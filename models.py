import torch
import math
import os


class MultiVAE(torch.nn.Module):
    def __init__(self, in_blocks, in_shared, out_shared, out_blocks, init="xavier"):
        super(MultiVAE, self).__init__()
        self.in_models = []
        self.in_slice = []
        self.out_slice = []
        for block in in_blocks:
            mlp = MLP(block, init=init)
            last_idx = str(len(mlp.layers))
            mlp.layers.add_module(last_idx, torch.nn.ReLU())
            self.in_models.append(mlp)
            self.in_slice.append(block[0])
            self.out_slice.append(block[-1])
        self.in_models = torch.nn.ModuleList(self.in_models)

        self.out_models = []
        for block in out_blocks:
            mlp = MLP(block, init=init)
            self.out_models.append(mlp)
        self.out_models = torch.nn.ModuleList(self.out_models)

        self.encoder = MLP(in_shared, init=init)
        self.decoder = MLP(out_shared, init=init)
        self.z_d = in_shared[-1]//2
        self.prior = torch.distributions.Normal(torch.tensor(0.0), torch.tensor(1.0))

    def forward(self, x, sample=True):
        begin = 0
        outs = []
        for i, model in enumerate(self.in_models):
            partial_x = x[:, begin:(begin+self.in_slice[i])]
            outs.append(model(partial_x))
            begin += self.in_slice[i]

        h = torch.cat(outs, dim=1)
        h = self.encoder(h)
        mu, logstd = h[:, :self.z_d], h[:, self.z_d:]
        std = torch.exp(logstd)
        if sample:
            dist = torch.distributions.Normal(mu, std)
            z = dist.rsample()
        else:
            z = mu
        h = self.decoder(z)

        begin = 0
        out_mu = []
        out_logstd = []
        for i, model in enumerate(self.out_models):
            partial_h = h[:, begin:(begin+self.out_slice[i])]
            out = model(partial_h)
            d = out.shape[1]//2
            out_mu.append(out[:, :d])
            out_logstd.append(out[:, d:])
            begin += self.out_slice[i]
        out_mu = torch.cat(out_mu, dim=1)
        out_logstd = torch.cat(out_logstd, dim=1)
        return mu, logstd, out_mu, out_logstd

    def loss(self, x, sample=True):
        z_mu, z_logstd, o_mu, o_logstd = self.forward(x, sample)
        z_std = torch.exp(z_logstd)
        z_dist = torch.distributions.Normal(z_mu, z_std)
        kl_loss = torch.distributions.kl_divergence(z_dist, self.prior).sum(dim=1).mean()

        o_std = torch.exp(o_logstd)
        o_dist = torch.distributions.Normal(o_mu, o_std)
        recon_loss = -o_dist.log_prob(x).sum(dim=1).mean()
        return recon_loss + kl_loss


class MLP(torch.nn.Module):
    def __init__(self, layer_info, activation="relu", init="he"):
        super(MLP, self).__init__()

        if activation == "relu":
            func = torch.nn.ReLU()
        elif activation == "tanh":
            func = torch.nn.Tanh()

        if init == "xavier":
            gain = 1.0
        else:
            gain = torch.nn.init.calculate_gain(activation)

        layers = []
        in_dim = layer_info[0]
        for i, unit in enumerate(layer_info[1:-1]):
            layers.append(Linear(in_features=in_dim, out_features=unit, gain=gain, init=init))
            layers.append(func)
            in_dim = unit
        layers.append(Linear(in_features=in_dim, out_features=layer_info[-1], gain=1.0, init=init))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    def load(self, path, name):
        state_dict = torch.load(os.path.join(path, name+".ckpt"))
        self.load_state_dict(state_dict)

    def save(self, path, name):
        dv = self.layers[-1].weight.device
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.cpu().state_dict(), os.path.join(path, name+".ckpt"))
        self.train().to(dv)


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, gain=1.0, init="he"):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.Tensor(out_features))

        if init == "he":
            d = (self.weight.size(1))
        elif init == "xavier":
            d = (self.weight.size(0) + self.weight.size(1)) / 2
        stdv = gain * math.sqrt(1 / d)
        self.weight.data.normal_(0., stdv)
        self.bias.data.zero_()

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight, self.bias)

    def extra_repr(self):
        return "in_features={}, out_features={}".format(self.in_features, self.out_features)
