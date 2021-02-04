"""Test multimodal VAE for UR10 data."""
import os
import argparse
import yaml
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import models
import data
import utils

parser = argparse.ArgumentParser("Test multimodal VAE.")
parser.add_argument("-opts", help="option file", type=str, required=True)
parser.add_argument("-banned", help="banned modalities", nargs="+", type=int, required=True)
parser.add_argument("-prefix", help="output prefix", type=str, required=True)
args = parser.parse_args()

opts = yaml.safe_load(open(args.opts, "r"))
print(yaml.dump(opts))

testset = data.MyDataset("data", modality=["endpoint", "joint"], action=["grasp", "move"], mode="test")

model = models.MultiVAE(
    in_blocks=opts["in_blocks"],
    in_shared=opts["in_shared"],
    out_shared=opts["out_shared"],
    out_blocks=opts["out_blocks"],
    init=opts["init_method"])
model.to(opts["device"])
model.load(opts["save"], "multivae_last")
model.cpu().eval()
print(model)

out_folder = os.path.join(opts["save"], "outs")
yje = torch.zeros(7)
zje = torch.zeros(7)
ype = torch.zeros(7)
zpe = torch.zeros(7)
N = 10

condition_idx = [68, 30, 60, 35, 72, 45, 61, 35, 43, 48]

for exp in range(N):
    exp_folder = os.path.join(out_folder, str(exp))
    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)

    x_pos, x_joint = testset.get_trajectory(exp)
    L = x_pos.shape[0]
    start_idx = condition_idx[exp]
    forward_t = L - start_idx - 1
    backward_t = start_idx
    x_all = [x_pos, x_joint]
    xn_pos, xn_joint = utils.noise_input(x_all, args.banned, prob=[0., 1.], direction="forward", modality_noise=False)
    x_condition = [x_pos[start_idx:(start_idx+1)], x_joint[start_idx:(start_idx+1)]]
    x_condition[0][:, 7:] = x_condition[0][:, :7]
    x_condition[1][:, 7:] = x_condition[1][:, :7]

    with torch.no_grad():
        # one-step forward prediction
        _, _, (y_pos, y_joint), _ = model([xn_pos, xn_joint], sample=False)
        # forecasting
        z_pos, z_joint = model.forecast(x_condition, forward_t, backward_t, banned_modality=args.banned)

        y_pos.clamp_(-1., 1.)
        y_joint.clamp_(-1., 1.)
        z_pos.clamp_(-1., 1.)
        z_joint.clamp_(-1, 1.)

        fig, ax = plt.subplots(3, 2, figsize=(12, 10))
        for i in range(3):
            for j in range(2):
                ax[i][j].plot(x_joint[:, i*2 + j] * 3, c="k")
                ax[i][j].plot(y_joint[:, i*2 + j] * 3, c="b")
                ax[i][j].plot(z_joint[:, i*2 + j] * 3, c="m")
                ax[i][j].scatter(start_idx, x_joint[start_idx, i*2 + j] * 3, c="r", marker="x")
                ax[i][j].set_ylabel("$q_%d$" % (i*2+j))
                ax[i][j].set_xlabel("Timesteps")
        pp = PdfPages(os.path.join(exp_folder, args.prefix+"-joints.pdf"))
        pp.savefig(fig)
        pp.close()
        plt.close()

        yje += ((x_joint[:, :7] - y_joint[:, :7])*3).abs().mean(dim=0)
        zje += ((x_joint[:, :7] - z_joint[:, :7])*3).abs().mean(dim=0)

        fig, ax = plt.subplots(3, 2, figsize=(12, 10))
        mapper = ["x", "y", "z", "rx", "ry", "rz"]
        for i in range(2):
            for j in range(3):
                ax[j][i].plot(x_pos[:, i*3 + j] * 3, c="k")
                ax[j][i].plot(y_pos[:, i*3 + j] * 3, c="b")
                ax[j][i].plot(z_pos[:, i*3 + j] * 3, c="m")
                ax[j][i].scatter(start_idx, x_pos[start_idx, i*3 + j] * 3, c="r", marker="x")
                ax[j][i].set_ylabel("$%s$" % mapper[i*3+j])
                ax[j][i].set_xlabel("Timesteps")
        pp = PdfPages(os.path.join(exp_folder, args.prefix+"-pos.pdf"))
        pp.savefig(fig)
        pp.close()
        plt.close()
        ype += ((x_pos[:, :7] - y_pos[:, :7])*3).abs().mean(dim=0)
        zpe += ((x_pos[:, :7] - z_pos[:, :7])*3).abs().mean(dim=0)

yje = np.degrees(yje/N)
zje = np.degrees(zje/N)
ype = (ype/N)
zpe = (zpe/N)
print("onestep joint errors: %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f" % (yje[0], yje[1], yje[2], yje[3], yje[4], yje[5], np.radians(yje[6])/30), file=open(os.path.join(out_folder, args.prefix+"-result.txt"), "a"))
print("forecast joint errors: %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f" % (zje[0], zje[1], zje[2], zje[3], zje[4], zje[5], np.radians(zje[6])/30), file=open(os.path.join(out_folder, args.prefix+"-result.txt"), "a"))
print("onestep position errors: %.4f, %.4f, %.4f, %.4f, %.4f, %.4f" % (ype[0], ype[1], ype[2], ype[3], ype[4], ype[5]), file=open(os.path.join(out_folder, args.prefix+"-result.txt"), "a"))
print("forecast position errors: %.4f, %.4f, %.4f, %.4f, %.4f, %.4f" % (zpe[0], zpe[1], zpe[2], zpe[3], zpe[4], zpe[5]), file=open(os.path.join(out_folder, args.prefix+"-result.txt"), "a"))
