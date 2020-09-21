"""Train multimodal VAE."""
import os
import time
import argparse
import yaml
import torch
from torch.utils.tensorboard import SummaryWriter
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

idx = torch.randperm(40)[:opts["traj_count"]].tolist()
val_cnd = [71, 40, 67, 56, 58, 56, 50, 79, 50, 53]

trainset = data.UR10Dataset("data", modality=["img", "joint"], action=["grasp", "move"], mode="train", traj_list=idx)
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
best_error = 1e5

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

    if mse_val_noised < best_error:
        best_error = mse_val_noised.item()
        model.save(opts["save"], "multivae_best")

    model.save(opts["save"], "multivae_last")
