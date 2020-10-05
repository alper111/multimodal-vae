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
parser.add_argument("-banned", help="banned modalities", nargs="+", type=int, required=True)
parser.add_argument("-prefix", help="output prefix", type=str, required=True)
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
model.load(opts["save"], "multivae_best")
model.cpu().eval()
print(model)

out_folder = os.path.join(opts["save"], "outs")
yje = torch.zeros(7)
zje = torch.zeros(7)
ype = 0.0
zpe = 0.0
N = 10

condition_idx = [68, 30, 60, 35, 72, 45, 61, 35, 43, 48]

for exp in range(N):
    exp_folder = os.path.join(out_folder, str(exp))
    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)

    x_img, x_joint = testset.get_trajectory(exp)
    L = x_img.shape[0]
    start_idx = condition_idx[exp]
    forward_t = L - start_idx - 1
    backward_t = start_idx
    x_all = [x_img, x_joint]
    xn_img, xn_joint = utils.noise_input(x_all, args.banned, prob=[0., 1.], direction="forward", modality_noise=False)
    x_condition = [x_img[start_idx:(start_idx+1)], x_joint[start_idx:(start_idx+1)]]
    # x[t+1] <- x[t]
    x_condition[0][:, 3:] = x_condition[0][:, :3]
    x_condition[1][:, 7:] = x_condition[1][:, :7]

    with torch.no_grad():
        # one-step forward prediction
        _, _, (y_img, y_joint), _ = model([xn_img, xn_joint], sample=False)
        # forecasting
        z_img, z_joint = model.forecast(x_condition, forward_t, backward_t, banned_modality=args.banned)

        y_img.clamp_(-1., 1.)
        y_joint.clamp_(-1., 1.)
        z_img.clamp_(-1., 1.)
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

        yje += ((x_joint[:, :7] - y_joint[:, :7])*3).abs().mean(dim=0)
        zje += ((x_joint[:, :7] - z_joint[:, :7])*3).abs().mean(dim=0)

        y_pixel_error = (utils.to_pixel(x_img[:, :3]) - utils.to_pixel(y_img[:, :3])).abs().mean()
        z_pixel_error = (utils.to_pixel(x_img[:, :3]) - utils.to_pixel(z_img[:, :3])).abs().mean()
        ype += y_pixel_error
        zpe += z_pixel_error
        print("%.4f, %.4f" % (y_pixel_error, z_pixel_error), file=open(os.path.join(out_folder, args.prefix+"-result.txt"), "a"))

        x_cat = torch.cat([x_img[:, :3], y_img[:, :3]], dim=3).permute(0, 2, 3, 1)
        torchvision.io.write_video(os.path.join(exp_folder, args.prefix+"-onestep.mp4"), utils.to_pixel(x_cat).byte(), fps=30)
        x_cat = torch.cat([x_img[:, :3], z_img[:, :3]], dim=3).permute(0, 2, 3, 1)
        torchvision.io.write_video(os.path.join(exp_folder, args.prefix+"-forecast.mp4"), utils.to_pixel(x_cat).byte(), fps=30)
        x_diff = (utils.to_pixel(x_img[:, :3]) - utils.to_pixel(z_img[:, :3])).abs().permute(0, 2, 3, 1)
        torchvision.io.write_video(os.path.join(exp_folder, args.prefix+"-diff.mp4"), x_diff, fps=30)

yje = np.degrees(yje/N)
zje = np.degrees(zje/N)
ype = ype / N
zpe = zpe / N

print("%.4f" % ype, file=open(os.path.join(out_folder, args.prefix+"-result.txt"), "a"))
print("%.4f" % zpe, file=open(os.path.join(out_folder, args.prefix+"-result.txt"), "a"))
print("%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f" % (yje[0], yje[1], yje[2], yje[3], yje[4], yje[5], np.radians(yje[6])/30), file=open(os.path.join(out_folder, args.prefix+"-result.txt"), "a"))
print("%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f" % (zje[0], zje[1], zje[2], zje[3], zje[4], zje[5], np.radians(zje[6])/30), file=open(os.path.join(out_folder, args.prefix+"-result.txt"), "a"))
