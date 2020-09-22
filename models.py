"""MultiVAE and other models."""
import torch
import math
import os
import utils


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
                conv = ConvEnc(block[3:], block[1], block[2])
                last_idx = str(len(conv.conv))
                conv.conv.add_module(last_idx, torch.nn.ReLU())

                # conv = torch.nn.Sequential(ConvEncoder(block[4:], [6, 128, 128], 128), torch.nn.ReLU())
                self.in_models.append(conv)
        self.in_models = torch.nn.ModuleList(self.in_models)

        self.out_models = []
        for block in out_blocks:
            if block[0] == -1:
                mlp = MLP(block[1:], init=init)
                self.out_models.append(mlp)
            else:
                conv = ConvDec(block[3:], block[1], block[2])
                # conv = ConvDecoder(block[4:], [512, 4, 4], 128)
                self.out_models.append(conv)
        self.out_models = torch.nn.ModuleList(self.out_models)

        self.encoder = torch.nn.Sequential(MLP(in_shared, init=init), torch.nn.Tanh())
        self.decoder = torch.nn.Sequential(MLP(out_shared, init=init), torch.nn.ReLU())
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

    def loss(self, x, y, lambd=1.0, beta=1.0, sample=True, reduce=True, mse=False):
        """
        Compute ELBO.

        Parameters
        ----------
        x : list of torch.Tensor
            Prediction tensor.
        y : list of torch.Tensor
            Target tensor.
        lambd : float, optional
            Coefficient of reconstruction loss.
        beta : float, optional
            Coefficient of KL divergence.
        sample : bool, optional
            In forward pass, sampling is done from the latent distribution if
            set true. Otherwise mean of the distribution is used.
        reduce : bool, optional
            If set to true, the loss is summed across the feature dimension.
            Otherwise, the loss is averaged across the feature dimension.
        mse : bool, optional
            If set to true, the original VAE loss is used in reconstruction
            (i.e. std=1.0). Otherwise, log-std is also predicted and NLL is
            calculated and minimized with the predicted std.

        Returns
        -------
        loss : torch.Tensor
            Total loss (evidence lower bound).
        """
        z_mu, z_logstd, o_mu, o_logstd = self.forward(x, sample)
        z_std = torch.exp(z_logstd)
        z_dist = torch.distributions.Normal(z_mu, z_std)
        kl_loss = torch.distributions.kl_divergence(z_dist, self.prior)
        if reduce:
            kl_loss = kl_loss.mean()
        else:
            kl_loss = kl_loss.sum(dim=1).mean()

        recon_loss = 0.0
        for x_m, x_s, y_m in zip(o_mu, o_logstd, y):
            x_m = x_m.reshape(x_m.shape[0], -1)
            x_s = x_s.reshape(x_s.shape[0], -1)
            y_m = y_m.reshape(y_m.shape[0], -1)
            if mse:
                modal_loss = torch.nn.functional.mse_loss(x_m, y_m, reduction="none")
            else:
                x_std = torch.exp(x_s)
                x_dist = torch.distributions.Normal(x_m, x_std)
                modal_loss = -x_dist.log_prob(y_m)

            if reduce:
                recon_loss += modal_loss.mean()
            else:
                recon_loss += modal_loss.sum(dim=1).mean()

        recon_loss /= len(y)
        loss = lambd * recon_loss + beta * kl_loss
        return loss

    def forecast(self, x, forward_t, backward_t, banned_modality):
        D = len(x)
        dims = []
        trajectory = []
        for i in range(D):
            trajectory.append([x[i].clone()])
            dims.append(x[i].shape[1] // 2)

        with torch.no_grad():

            for t in range(forward_t):
                x_noised = utils.noise_input(x, banned_modality=banned_modality, prob=[0.0, 1.0], direction="forward")
                _, _, x, _ = self.forward(x_noised, sample=False)
                for i in range(D):
                    x[i].clamp_(-1.0, 1.0)
                    trajectory[i].append(x[i].clone())
                    # x[t] <- x[t+1]
                    x[i][:, :dims[i]] = x[i][:, dims[i]:]

            x = [trajectory[0][0], trajectory[1][0]]
            for _ in range(backward_t):
                x_noised = utils.noise_input(x, banned_modality=banned_modality, prob=[0.0, 1.0], direction="backward")
                _, _, x, _ = self.forward(x_noised, sample=False)
                for i in range(D):
                    x[i].clamp_(-1.0, 1.0)
                    trajectory[i].insert(0, x[i].clone())
                    # x[t+1] <- x[t]
                    x[i][:, dims[i]:] = x[i][:, :dims[i]]

            # interpolation on conditioned point
            for i in range(D):
                back_pred = trajectory[i][backward_t-1][:, dims[i]:]
                forward_pred = trajectory[i][backward_t+1][:, :dims[i]]
                trajectory[i][backward_t][:, :dims[i]] = (back_pred+forward_pred)/2
                trajectory[i][backward_t][:, dims[i]:] = (back_pred+forward_pred)/2

            # x[t] <- x[t+1] for forward steps
            for t in range(forward_t):
                for i in range(D):
                    trajectory[i][backward_t+1][:, :dims[i]] = trajectory[i][backward_t+1][:, dims[i]:]

        for i in range(D):
            trajectory[i] = torch.cat(trajectory[i], dim=0)
        return trajectory

    def load(self, path, name):
        state_dict = torch.load(os.path.join(path, name+".ckpt"))
        self.load_state_dict(state_dict)

    def save(self, path, name):
        dv = self.encoder[0].layers[-1].weight.device
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


class ConvEnc(torch.nn.Module):
    def __init__(self, channel_list, hidden_dim, out_dim):
        super(ConvEnc, self).__init__()
        convs = []
        for i in range(len(channel_list)-1):
            convs.append(Conv3x3Block(channel_list[i], channel_list[i+1], sampling="down"))
        self.conv = torch.nn.Sequential(
            torch.nn.Sequential(*convs),
            Flatten([1, 2, 3]),
            torch.nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        return self.conv(x)


class ConvDec(torch.nn.Module):
    def __init__(self, channel_list, in_dim, hidden_dim):
        super(ConvDec, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU())
        self.conv = []
        for i in range(len(channel_list)-1):
            self.conv.append(Conv3x3Block(channel_list[i], channel_list[i+1], sampling="up"))

        self.conv.append(torch.nn.Conv2d(channel_list[-1], 16, kernel_size=3, stride=1, padding=1))
        self.conv.append(torch.nn.ReLU())
        self.conv.append(torch.nn.Conv2d(16, 12, kernel_size=3, stride=1, padding=1))
        self.conv.append(torch.nn.ReLU())
        self.conv.append(torch.nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1))
        # self.conv.append(torch.nn.Tanh())
        self.conv = torch.nn.Sequential(*self.conv)
        self.first_filter = self.conv[0].conv[0].weight.shape[1]

    def forward(self, x):
        h = self.fc(x)
        width = int(math.sqrt(h.shape.numel() // (x.shape[0] * self.first_filter)))
        h = h.reshape(x.shape[0], self.first_filter, width, width)
        return self.conv(h)


class Conv3x3Block(torch.nn.Module):
    def __init__(self, in_channel, out_channel, sampling):
        super(Conv3x3Block, self).__init__()
        self.conv = [torch.nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)]
        self.conv.append(torch.nn.ReLU())
        if sampling == "down":
            self.conv.append(torch.nn.MaxPool2d(kernel_size=2))
        elif sampling == "up":
            self.conv.append(torch.nn.Upsample(scale_factor=2, mode="nearest"))
        self.conv = torch.nn.Sequential(*self.conv)

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


class ConvEncoder(torch.nn.Module):
    def __init__(self, channels, input_shape, latent_dim, activation=torch.nn.ReLU()):
        super(ConvEncoder, self).__init__()
        convolutions = []
        current_shape = input_shape
        for ch in channels:
            convolutions.append(torch.nn.Conv2d(in_channels=current_shape[0], out_channels=ch, kernel_size=4, stride=2, padding=1))
            current_shape = [ch, current_shape[1] // 2, current_shape[2] // 2]
            convolutions.append(activation)
        self.convolutions = torch.nn.Sequential(*convolutions)
        self.dense = torch.nn.Linear(in_features=current_shape[0] * current_shape[1] * current_shape[2], out_features=latent_dim)

    def forward(self, x, y=None):
        out = self.convolutions(x)
        out = out.view(out.shape[0], -1)
        out = self.dense(out)
        return out


class ConvDecoder(torch.nn.Module):
    def __init__(self, channels, input_shape, latent_dim, activation=torch.nn.ReLU()):
        super(ConvDecoder, self).__init__()
        self.input_shape = input_shape
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(in_features=latent_dim, out_features=input_shape[0] * input_shape[1] * input_shape[2]),
            activation)

        convolutions = []
        current_shape = input_shape
        for ch in channels[:-1]:
            convolutions.append(torch.nn.ConvTranspose2d(in_channels=current_shape[0], out_channels=ch, kernel_size=4, stride=2, padding=1))
            current_shape = [ch, current_shape[1] * 2, current_shape[2] * 2]
            convolutions.append(activation)
        convolutions.append(torch.nn.ConvTranspose2d(in_channels=current_shape[0], out_channels=channels[-1], kernel_size=4, stride=2, padding=1))
        self.convolutions = torch.nn.Sequential(*convolutions)

    def forward(self, x):
        out = self.dense(x)
        out = out.reshape(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        out = self.convolutions(out)
        return out
