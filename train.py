"""Train multimodal VAE."""
import os
import time
import argparse
import yaml
import numpy as np
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import models
import data
import utils


parser = argparse.ArgumentParser("Train multimodal VAE.")
parser.add_argument("-opts", help="option file", type=str, required=True)
args = parser.parse_args()

opts = yaml.safe_load(open(args.opts, "r"))
if not os.path.exists(opts["save"]):
    os.makedirs(opts["save"])
opts["time"] = time.asctime(time.localtime(time.time()))
dev = opts["device"]
file = open(os.path.join(opts["save"], "opts.yaml"), "w")
yaml.dump(opts, file)
file.close()
print(yaml.dump(opts))

logdir = os.path.join(opts["save"], "log")
writer = SummaryWriter(logdir)

trainset = data.UR10Dataset("data", modality=["img", "joint"], action=["grasp", "move"], mode="train")
valset = data.UR10Dataset("data", modality=["img", "joint"], action=["grasp", "move"], mode="val")

trainloader = torch.utils.data.DataLoader(trainset, batch_size=opts["batch_size"], shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=10000, shuffle=False)
x_val_img, x_val_joint = iter(valloader).next()
x_val_img = x_val_img.to(dev)
x_val_joint = x_val_joint.to(dev)
x_val_all = [x_val_img, x_val_joint]

model = models.MultiVAE(
    in_blocks=opts["in_blocks"],
    in_shared=opts["in_shared"],
    out_shared=opts["out_shared"],
    out_blocks=opts["out_blocks"],
    init=opts["init_method"])
model.to(dev)
optimizer = torch.optim.Adam(lr=opts["lr"], params=model.parameters(), amsgrad=True)
print(model)
print("Parameter count:", utils.get_parameter_count(model))

for e in range(opts["epoch"]):
    running_avg = 0.0
    for i, (x_img, x_joint) in enumerate(trainloader):
        optimizer.zero_grad()
        x_img = x_img.to(dev)
        x_joint = x_joint.to(dev)

        x_all = [x_img, x_joint]
        x_noised = utils.noise_input(x_all, banned_modality=[0, 0], prob=[0.5, 0.5, 0.5],
                                     direction="both", modality_noise=True)

        loss = model.mse_loss(x_noised, x_all, lambd=opts["lambda"], beta=opts["beta"], sample=False, reduce=True)
        loss.backward()
        optimizer.step()
        running_avg += loss.item()

        del x_all[:], x_noised[:]

    running_avg /= (i+1)
    with torch.no_grad():
        x_val_plain = utils.noise_input(x_val_all, banned_modality=[0, 0], prob=[1.0, 0.0])
        x_val_noised = utils.noise_input(x_val_all, banned_modality=[0, 0], prob=[0.0, 0.5, 0.5],
                                         direction="both", modality_noise=True)
        mse_val = model.mse_loss(x_val_plain, x_val_all, sample=False, beta=0.0, reduce=True)
        mse_val_noised = model.mse_loss(x_val_noised, x_val_all, sample=False, beta=0.0, reduce=True)

    del x_val_plain[:], x_val_noised[:]

    writer.add_scalar("Epoch loss", running_avg, e)
    writer.add_scalar("MSE val",  mse_val, e)
    writer.add_scalar("MSE val noised",  mse_val_noised, e)
    print("Epoch %d loss: %.5f, MSE val: %.5f, Noised: %.5f" % (e+1, running_avg, mse_val, mse_val_noised))

    if (e+1) % 20 == 0:
        model.save(opts["save"], "multivae_%d" % (e+1))
        with torch.no_grad():
            x_img, x_joint = valset.get_trajectory(np.random.randint(0, 1))
            N = x_img.shape[0]
            start_idx = int(0.5 * N)
            forward_t = N - start_idx - 1
            backward_t = start_idx
            x_all = [x_img.to(dev), x_joint.to(dev)]
            xn_img, xn_joint = utils.noise_input(x_all, banned_modality=[0, 0], prob=[0., 1.], direction="forward")
            x_condition = [x_img[start_idx:(start_idx+1)].to(dev), x_joint[start_idx:(start_idx+1)].to(dev)]
            x_condition[0][:, :3] = x_condition[0][:, 3:]
            x_condition[1][:, :7] = x_condition[1][:, 7:]

            # one-step forward prediction
            _, _, (y_img, y_joint), _ = model([xn_img, xn_joint], sample=False)
            # forecasting
            z_img, z_joint = model.forecast(x_condition, forward_t, backward_t, banned_modality=[0, 0])

            y_img.clamp_(-1., 1.)
            y_joint.clamp_(-1., 1.)
            z_img.clamp_(-1., 1.)
            z_joint.clamp_(-1, 1.)

        fig, ax = plt.subplots(3, 2)
        for i in range(3):
            for j in range(2):
                ax[i][j].plot(x_joint[:, i*2 + j + 7], c="k")
                ax[i][j].plot(y_joint[:, i*2 + j + 7].cpu(), c="b")
                ax[i][j].plot(z_joint[:, i*2 + j + 7].cpu(), c="m")
                ax[i][j].scatter(start_idx, x_joint[start_idx, i*2 + j + 7], c="r", marker="x")
                ax[i][j].set_ylabel("$q_%d$" % (i*2+j))
                ax[i][j].set_xlabel("Timesteps")
        pp = PdfPages(os.path.join(opts["save"], "joints_recons%d.pdf" % (e+1)))
        pp.savefig(fig)
        pp.close()

        x_cat = torch.cat([x_img[:, 3:], y_img[:, 3:].cpu()], dim=3).permute(0, 2, 3, 1)
        torchvision.io.write_video(os.path.join(opts["save"], "onestep_%d.mp4" % (e+1)),
                                   utils.to_pixel(x_cat).byte(), fps=30)
        x_cat = torch.cat([x_img[:, 3:], z_img[:, 3:].cpu()], dim=3).permute(0, 2, 3, 1)
        torchvision.io.write_video(os.path.join(opts["save"], "forecast_%d.mp4" % (e+1)),
                                   utils.to_pixel(x_cat).byte(), fps=30)

        del x_all[:], x_condition[:], z_img, z_joint

    model.save(opts["save"], "multivae_last")
