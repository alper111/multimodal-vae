"""Test multimodal VAE for UR10 data."""
import os
import argparse
import yaml
import torch
import torchvision
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import models
import data
import utils

parser = argparse.ArgumentParser("Test multimodal VAE.")
parser.add_argument("-opts", help="option file", type=str, required=True)
args = parser.parse_args()

opts = yaml.safe_load(open(args.opts, "r"))
print(yaml.dump(opts))

testset = data.UR10Dataset("data", modality=["img", "joint"], action=["grasp", "move"], mode="test")

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
banned_mods = [0, 0]
yje = torch.zeros(6)
zje = torch.zeros(6)

for exp in range(20):
    exp_folder = os.path.join(out_folder, str(exp))
    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)

    x_img, x_joint = testset.get_trajectory(exp)
    N = x_img.shape[0]
    start_idx = int(0.5 * N)
    forward_t = N - start_idx - 1
    backward_t = start_idx
    x_all = [x_img, x_joint]
    xn_img, xn_joint = utils.noise_input(x_all, banned_mods, prob=[0., 1.], direction="forward", modality_noise=False)
    x_condition = [x_img[start_idx:(start_idx+1)], x_joint[start_idx:(start_idx+1)]]
    x_condition[0][:, :3] = x_condition[0][:, 3:]
    x_condition[1][:, :6] = x_condition[1][:, 6:]

    with torch.no_grad():
        # one-step forward prediction
        _, _, (y_img, y_joint), _ = model([xn_img, xn_joint], sample=False)
        # forecasting
        z_img, z_joint = model.forecast(x_condition, forward_t, backward_t, banned_modality=banned_mods)

        y_img.clamp_(-1., 1.)
        y_joint.clamp_(-1., 1.)
        z_img.clamp_(-1., 1.)
        z_joint.clamp_(-1, 1.)

        fig, ax = plt.subplots(3, 2, figsize=(12, 10))
        for i in range(3):
            for j in range(2):
                ax[i][j].plot(x_joint[:, i*2 + j + 6], c="k")
                ax[i][j].plot(y_joint[:, i*2 + j + 6], c="b")
                ax[i][j].plot(z_joint[:, i*2 + j + 6], c="m")
                ax[i][j].scatter(start_idx, x_joint[start_idx, i*2 + j + 6], c="r", marker="x")
                ax[i][j].set_ylabel("$q_%d$" % (i*2+j))
                ax[i][j].set_xlabel("Timesteps")
        pp = PdfPages(os.path.join(exp_folder, "both-joints.pdf"))
        pp.savefig(fig)
        pp.close()

        yje += ((x_joint[:, 6:] - y_joint[:, 6:])*3).abs().mean(dim=0)
        zje += ((x_joint[:, 6:] - z_joint[:, 6:])*3).abs().mean(dim=0)

        y_pixel_error = (utils.to_pixel(x_img[:, 3:]) - utils.to_pixel(y_img[:, 3:])).abs().mean()
        z_pixel_error = (utils.to_pixel(x_img[:, 3:]) - utils.to_pixel(z_img[:, 3:])).abs().mean()
        print("Exp: %d, onestep pixel error: %.4f, forecast pixel error: %.4f" % (exp, y_pixel_error, z_pixel_error))

        x_cat = torch.cat([x_img[:, 3:], y_img[:, 3:]], dim=3).permute(0, 2, 3, 1)
        torchvision.io.write_video(os.path.join(exp_folder, "both-onestep.mp4"), utils.to_pixel(x_cat).byte(), fps=30)
        x_cat = torch.cat([x_img[:, 3:], z_img[:, 3:]], dim=3).permute(0, 2, 3, 1)
        torchvision.io.write_video(os.path.join(exp_folder, "both-forecast.mp4"), utils.to_pixel(x_cat).byte(), fps=30)

yje = np.degrees(yje/20)
zje = np.degrees(zje/20)
print("onestep joint errors: %.4f, %.4f, %.4f, %.4f, %.4f, %.4f" % (yje[0], yje[1], yje[2], yje[3], yje[4], yje[5]))
print("forecast joint errors: %.4f, %.4f, %.4f, %.4f, %.4f, %.4f" % (zje[0], zje[1], zje[2], zje[3], zje[4], zje[5]))
