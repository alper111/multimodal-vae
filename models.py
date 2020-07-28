"""MultiVAE and other models."""
import torch
import math
import os


class MultiVAE(torch.nn.Module):
    """Multimodal variational autoencoder."""

    def __init__(self, in_blocks, in_shared, out_shared, out_blocks, init="xavier"):
        """Initialize a multimodal VAE instance.

        Parameters
        ----------
        in_blocks : list of list of int
            Encoder networks for different modalities. If a list starts with
            -1, an MLP is created with hidden units specified in the list. If
            a list starts with -2, a CNN is created with channel numbers
            specified in the list. For CNN, list[1] and list[2] represent
            the input and the output dimensions for the fully-connected layer.
        in_shared : list of int
            Shared encoder's units (including input layer).
        out_shared : list of int
            Shared decoder's units (including input layer).
        out_blocks : list of list of int
            Decoder networks for different modalities. If a list starts with
            -1, an MLP is created with hidden units specified in the list. If
            a list starts with -2, a CNN is created with channel numbers
            specified in the list. Convolutions are transposed. For CNN, list[1]
            and list[2] represent the input and the output dimensions for the
            fully-connected layer. As decoders predict mean and standard
            deviation, output dimension should be doubled.
        init : "xavier" or "he"
            Initialization technique.
        """
        super(MultiVAE, self).__init__()
        self.in_models = []
        for block in in_blocks:
            if block[0] == -1:
                mlp = MLP(block[1:], init=init)
                last_idx = str(len(mlp.layers))
                mlp.layers.add_module(last_idx, torch.nn.ReLU())
                self.in_models.append(mlp)
            else:
                conv = ConvSeq(block[3:], block[1], block[2])
                last_idx = str(len(conv.conv))
                conv.conv.add_module(last_idx, torch.nn.ReLU())
                self.in_models.append(conv)
        self.in_models = torch.nn.ModuleList(self.in_models)

        self.out_models = []
        for block in out_blocks:
            if block[0] == -1:
                mlp = MLP(block[1:], init=init)
                self.out_models.append(mlp)
            else:
                conv = ConvTSeq(block[3:], block[1], block[2])
                self.out_models.append(conv)
        self.out_models = torch.nn.ModuleList(self.out_models)

        self.encoder = MLP(in_shared, init=init)
        self.decoder = MLP(out_shared, init=init)
        self.z_d = in_shared[-1]//2
        self.prior = torch.distributions.Normal(torch.tensor(0.0), torch.tensor(1.0))

    def forward(self, x, sample=True):
        """Forward pass.

        Parameters
        ----------
        x : list of torch.Tensor
            Input tensors for each modality.
        sample : bool
            Sampling is done from the latent distribution if set true.
            Otherwise mean of the distribution is used.

        Returns
        -------
        mu : list of torch.Tensor
            Mean of the latent distribution for each modality.
        logstd : torch.Tensor
            Log standard deviation of the latent distribution for each modality.
        out_mu : torch.Tensor
            Mean of the output distribution for each modality.
        out_logstd : torch.Tensor
            Log standard deviation of the output distribution for each modality.
        """
        outs = []
        out_slice = []
        for x_in, model in zip(x, self.in_models):
            o = model(x_in)
            outs.append(o)
            out_slice.append(o.shape[1])

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
            partial_h = h[:, begin:(begin+out_slice[i])]
            out = model(partial_h)
            d = out.shape[1]//2
            out_mu.append(out[:, :d])
            out_logstd.append(out[:, d:])
            begin += out_slice[i]
        return mu, logstd, out_mu, out_logstd

    def loss(self, x, y, sample=True, lambd=1.0, beta=1.0):
        """
        Compute ELBO.

        Parameters
        ----------
        x : list of torch.Tensor
            Prediction tensor.
        y : list of torch.Tensor
            Target tensor.
        sample : bool
            In forward pass, sampling is done from the latent distribution if
            set true. Otherwise mean of the distribution is used.
        lambd : float, optional
            Coefficient of reconstruction loss.
        beta : float, optional
            Coefficient of KL divergence.

        Returns
        -------
        loss : torch.Tensor
            Total loss (evidence lower bound).
        """
        z_mu, z_logstd, o_mu, o_logstd = self.forward(x, sample)
        z_std = torch.exp(z_logstd)
        z_dist = torch.distributions.Normal(z_mu, z_std)
        kl_loss = torch.distributions.kl_divergence(z_dist, self.prior).sum(dim=1).mean()

        recon_loss = 0.0
        for x_m, x_s, y_m in zip(o_mu, o_logstd, y):
            x_m = x_m.reshape(x_m.shape[0], -1)
            x_s = x_s.reshape(x_s.shape[0], -1)
            y_m = y_m.reshape(y_m.shape[0], -1)
            x_std = torch.exp(x_s)
            x_dist = torch.distributions.Normal(x_m, x_std)
            recon_loss += (-x_dist.log_prob(y_m).sum(dim=1))
        recon_loss = recon_loss.mean()
        loss = lambd * recon_loss + beta * kl_loss
        return loss

    def mse_loss(self, x, y, sample=True, lambd=1.0, beta=1.0, reduce=False):
        z_mu, z_logstd, o_mu, o_logstd = self.forward(x, sample)
        z_std = torch.exp(z_logstd)
        z_dist = torch.distributions.Normal(z_mu, z_std)
        kl_loss = torch.distributions.kl_divergence(z_dist, self.prior)
        if reduce:
            kl_loss = kl_loss.mean()
        else:
            kl_loss = kl_loss.sum(dim=1).mean()

        recon_loss = 0.0
        for x_m, y_m in zip(o_mu, y):
            x_m = x_m.reshape(x_m.shape[0], -1)
            y_m = y_m.reshape(y_m.shape[0], -1)
            if reduce:
                recon_loss += torch.nn.functional.mse_loss(x_m, y_m, reduction="none").mean()
            else:
                recon_loss += torch.nn.functional.mse_loss(x_m, y_m, reduction="none").sum(dim=1)
        recon_loss /= len(y)

        return lambd * recon_loss + beta * kl_loss

    def load(self, path, name):
        state_dict = torch.load(os.path.join(path, name+".ckpt"))
        self.load_state_dict(state_dict)

    def save(self, path, name):
        dv = self.encoder.layers[-1].weight.device
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.eval().cpu().state_dict(), os.path.join(path, name+".ckpt"))
        self.train().to(dv)


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
        torch.save(self.eval().cpu().state_dict(), os.path.join(path, name+".ckpt"))
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


class ConvSeq(torch.nn.Module):
    def __init__(self, channel_list, hidden_dim, out_dim):
        super(ConvSeq, self).__init__()
        convs = []
        for i in range(len(channel_list)-1):
            convs.append(Conv3x3Block(channel_list[i], channel_list[i+1]))
        self.conv = torch.nn.Sequential(
            torch.nn.Sequential(*convs),
            Flatten([1, 2, 3]),
            torch.nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        return self.conv(x)


class ConvTSeq(torch.nn.Module):
    def __init__(self, channel_list, in_dim, hidden_dim):
        super(ConvTSeq, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU())
        self.conv = []
        for i in range(len(channel_list)-1):
            self.conv.append(ConvT3x3Block(channel_list[i], channel_list[i+1]))
        self.conv = torch.nn.Sequential(*self.conv)
        self.first_filter = self.conv[0].conv[0].weight.shape[0]

    def forward(self, x):
        h = self.fc(x)
        width = int(math.sqrt(h.shape.numel() // (x.shape[0] * self.first_filter)))
        h = h.reshape(x.shape[0], self.first_filter, width, width)
        return self.conv(h)


class Conv3x3Block(torch.nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv3x3Block, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2))

    def forward(self, x):
        return self.conv(x)


class ConvT3x3Block(torch.nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ConvT3x3Block, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=2, mode="nearest"))

    def forward(self, x):
        return self.conv(x)


class Flatten(torch.nn.Module):
    def __init__(self, dims):
        super(Flatten, self).__init__()
        self.dims = dims

    def forward(self, x):
        dim = 1
        for d in self.dims:
            dim *= x.shape[d]
        return x.reshape(-1, dim)

    def extra_repr(self):
        return "dims=[" + ", ".join(list(map(str, self.dims))) + "]"
