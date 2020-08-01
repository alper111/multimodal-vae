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

parser = argparse.ArgumentParser("Train multimodal VAE.")
parser.add_argument("-opts", help="option file", type=str, required=True)
args = parser.parse_args()

opts = yaml.safe_load(open(args.opts, "r"))
if not os.path.exists(opts["save"]):
    os.makedirs(opts["save"])
opts["time"] = time.asctime(time.localtime(time.time()))
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
x_val_img = x_val_img.to(opts["device"])
x_val_joint = x_val_joint.to(opts["device"])

model = models.MultiVAE(
    in_blocks=opts["in_blocks"],
    in_shared=opts["in_shared"],
    out_shared=opts["out_shared"],
    out_blocks=opts["out_blocks"],
    init=opts["init_method"])
model.to(opts["device"])
optimizer = torch.optim.Adam(lr=opts["lr"], params=model.parameters(), amsgrad=True)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10, verbose=True)
print(model)

for e in range(opts["epoch"]):
    running_avg = 0.0
    for i, (x_img, x_joint) in enumerate(trainloader):
        optimizer.zero_grad()
        x_img = x_img.to(opts["device"])
        x_joint = x_joint.to(opts["device"])

        y_img, y_joint = x_img.clone(), x_joint.clone()
        loss = model.mse_loss([x_img, x_joint], [y_img, y_joint], lambd=opts["lambda"], beta=opts["beta"]*((0.95)**e), reduce=True)
        # loss = model.loss([x_img, x_joint], [y_img, y_joint], lambd=opts["lambda"], beta=opts["beta"])
        loss.backward()
        optimizer.step()
        running_avg += loss.item()
    running_avg /= (i+1)
    with torch.no_grad():
        y_val_img, y_val_joint = x_val_img.clone(), x_val_joint.clone()

        mse_val = model.mse_loss(
            [x_val_img, x_val_joint],
            [y_val_img, y_val_joint],
            sample=False, beta=0.0, reduce=True)

    # scheduler.step(mse_val)
    writer.add_scalar("Epoch loss", running_avg, e)
    writer.add_scalar("MSE val",  mse_val, e)
    print("Epoch %d loss: %.5f, MSE val: %.5f" % (e+1, running_avg, mse_val))

    if (e+1) % 20 == 0:
        model.save(opts["save"], "multivae_%d" % (e+1))
        with torch.no_grad():
            x_img, x_joint = valset.get_trajectory(np.random.randint(0, 16))
            _, _, o_mu, o_logstd = model([x_img.to(opts["device"]), x_joint.to(opts["device"])])
            img_m, joint_m = o_mu[0].cpu().clamp(-1., 1.), o_mu[1].cpu()
            img_s, joint_s = torch.exp(o_logstd[0]).cpu(), torch.exp(o_logstd[1]).cpu()
        pred_img = (((img_m[:, :3] / 2) + 0.5)*255).byte()
        orig_img = (((x_img[:, :3] / 2) + 0.5)*255).byte()
        img_cat = torch.cat([orig_img, pred_img], dim=3).permute(0, 2, 3, 1)
        torchvision.io.write_video(os.path.join(opts["save"], "recons_%d.mp4" % (e+1)), img_cat, fps=30)

        fig, ax = plt.subplots(3, 2)
        for i in range(3):
            for j in range(2):
                ax[i][j].plot(x_joint[:, i*2+j], c="k")
                ax[i][j].plot(joint_m[:, i*2+j], c="orange", linestyle="dashed")
                lower_plot = joint_m[:, i*2+j] - joint_s[:, i*2+j]
                upper_plot = joint_m[:, i*2+j] + joint_s[:, i*2+j]
                ax[i][j].fill_between(np.arange(x_joint.shape[0]), lower_plot, upper_plot, color="orange", alpha=0.5)
                ax[i][j].set_ylabel("$q_%d$" % (i*2+j))
                ax[i][j].set_xlabel("Timesteps")
                ax[i][j].set_ylim(-3, 3)
        pp = PdfPages(os.path.join(opts["save"], "joints_recons%d.pdf" % (e+1)))
        pp.savefig(fig)
        pp.close()

    model.save(opts["save"], "multivae_last")
