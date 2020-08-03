"""Test multimodal VAE for UR10 data."""
import argparse
import yaml
import torch
import torchvision
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import models
import data

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
model.eval()
print(model)

traj_idx = np.random.randint(0, 20)

with torch.no_grad():
    x_img, x_joint = testset.get_trajectory(traj_idx)
    N = x_img.shape[0]
    start_idx = int(0.2 * N)
    traj_length = N - start_idx
    x_all = [x_img.to(opts["device"]), x_joint.to(opts["device"])]
    x_partial = [x_img[start_idx:(start_idx+1)].to(opts["device"]), x_joint[start_idx:(start_idx+1)].to(opts["device"])]
    # reconstruction
    _, _, o_mu, o_logstd = model(x_all, sample=False)
    img_m, joint_m = o_mu[0].cpu().clamp(-1.0, 1.0), o_mu[1].cpu()
    img_s, joint_s = torch.exp(o_logstd[0]).cpu(), torch.exp(o_logstd[1]).cpu()
    # forecasting
    img_traj, joint_traj = model.forecast(x_partial, traj_length)

x_cat = torch.cat([x_img[:, :3], img_m[:, :3]], dim=3).permute(0, 2, 3, 1)
torchvision.io.write_video("out/img_recons.mp4", ((x_cat/2+0.5)*255).byte(), fps=30)
x_cat = torch.cat([x_img[start_idx:, 3:], img_traj[:, 3:].cpu()], dim=3).permute(0, 2, 3, 1)
torchvision.io.write_video("out/img_forecast.mp4", ((x_cat/2+0.5)*255).byte(), fps=30)

fig, ax = plt.subplots(3, 2)
for i in range(3):
    for j in range(2):
        ax[i][j].plot(x_joint[:, i*2+j], c="k")
        ax[i][j].plot(joint_m[:, i*2+j], c="orange", linestyle="dashed")
        ax[i][j].plot(torch.arange(start_idx, N), joint_traj[:, i*2+j].cpu(), c="m", linestyle="dashed")
        # lower_plot = joint_m[:, i*2+j] - joint_s[:, i*2+j]
        # upper_plot = joint_m[:, i*2+j] + joint_s[:, i*2+j]
        # ax[i][j].fill_between(np.arange(N), lower_plot, upper_plot, color="orange", alpha=0.5)
        ax[i][j].set_ylabel("$q_%d$" % (i*2+j))
        ax[i][j].set_xlabel("Timesteps")
        ax[i][j].set_ylim(-1, 1)
pp = PdfPages("out/joints_recons.pdf")
pp.savefig(fig)
pp.close()
