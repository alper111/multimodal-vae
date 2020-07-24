import os
import torch
import numpy as np
import models
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

run_name = "run-5"
out_path = os.path.join("out", run_name)
if not os.path.exists(out_path):
    os.makedirs(out_path)

x_test = torch.tensor(np.load("data/test.npy"), dtype=torch.float, device=device)

in_blocks = [
    [8, 40, 20],
    [8, 40, 20],
    [2, 10, 5],
    [2, 10, 5],
    [8, 40, 20]
]
out_blocks = [
    [20, 40, 8, 16],
    [20, 40, 8, 16],
    [5, 10, 2, 4],
    [5, 10, 2, 4],
    [20, 40, 8, 16]
]
in_shared = [70, 100, 56]
out_shared = [28, 100, 70]
model = models.MultiVAE(in_blocks=in_blocks, in_shared=in_shared, out_shared=out_shared, out_blocks=out_blocks, init="xavier")
model.to(device)
model.load(os.path.join("save", run_name), "multivae.pt")
model.eval()

x_prev = x_test.clone()
x_vis = x_test.clone()
x_prev[:, [0, 1, 2, 3, 8, 9, 10, 11, 16, 18, 20, 21, 22, 23]] = -2.0
x_vis[:, :8] = -2.0
x_vis[:, 16:] = -2.0

with torch.no_grad():
    _, _, y_prev_mu, y_prev_logstd = model(x_prev, sample=False)
    _, _, y_vis_mu, y_vis_logstd = model(x_vis, sample=False)
    y_prev_mu = y_prev_mu.cpu().numpy()
    y_vis_mu = y_vis_mu.cpu().numpy()
    y_prev_std = torch.exp(y_prev_logstd).cpu().numpy()
    y_vis_std = torch.exp(y_vis_logstd).cpu().numpy()

x_test = x_test.cpu().numpy()
fig, ax = plt.subplots(4, 1, figsize=(6, 6))
for i in range(4):
    ax[i].plot(x_test[:100, i], c="k")
    ax[i].plot(y_prev_mu[:100, i], c="b")
    ax[i].plot(y_vis_mu[:100, i], c="orange", linestyle="dashed")
    lower_plot = y_vis_mu[:100, i] - y_vis_std[:100, i]
    upper_plot = y_vis_mu[:100, i] + y_vis_std[:100, i]
    ax[i].fill_between(np.arange(100), lower_plot, upper_plot, color="orange", alpha=0.5)
    ax[i].set_ylabel("$q_%d$" % i)
    ax[i].set_xlabel("Timesteps")
pp = PdfPages(os.path.join(out_path, "q.pdf"))
pp.savefig(fig)
pp.close()
plt.close()

fig, ax = plt.subplots(4, 1, figsize=(6, 6))
for i in range(4):
    ax[i].plot(x_test[:100, i+8], c="k")
    ax[i].plot(y_prev_mu[:100, i+8], c="b")
    ax[i].plot(y_vis_mu[:100, i+8], c="orange", linestyle="dashed")
    lower_plot = y_vis_mu[:100, i+8] - y_vis_std[:100, i+8]
    upper_plot = y_vis_mu[:100, i+8] + y_vis_std[:100, i+8]
    ax[i].fill_between(np.arange(100), lower_plot, upper_plot, color="orange", alpha=0.5)
    ax[i].set_ylabel("$v_%d$" % i)
    ax[i].set_xlabel("Timesteps")
pp = PdfPages(os.path.join(out_path, "v.pdf"))
pp.savefig(fig)
pp.close()
plt.close()

fig, ax = plt.subplots(4, 1, figsize=(6, 6))
for i in range(4):
    ax[i].plot(x_test[:100, i+20], c="k")
    ax[i].plot(y_prev_mu[:100, i+20], c="b")
    ax[i].plot(y_vis_mu[:100, i+20], c="orange", linestyle="dashed")
    lower_plot = y_vis_mu[:100, i+20] - y_vis_std[:100, i+20]
    upper_plot = y_vis_mu[:100, i+20] + y_vis_std[:100, i+20]
    ax[i].fill_between(np.arange(100), lower_plot, upper_plot, color="orange", alpha=0.5)
    ax[i].set_ylabel("$u_%d$" % i)
    ax[i].set_xlabel("Timesteps")
pp = PdfPages(os.path.join(out_path, "u.pdf"))
pp.savefig(fig)
pp.close()
plt.close()

plt.figure(figsize=(6, 2))
plt.plot(x_test[:100, 16], c="k")
plt.plot(y_prev_mu[:100, 16], c="b")
plt.plot(y_vis_mu[:100, 16], c="orange", linestyle="dashed")
lower_plot = y_vis_mu[:100, 16] - y_vis_std[:100, 16]
upper_plot = y_vis_mu[:100, 16] + y_vis_std[:100, 16]
plt.fill_between(np.arange(100), lower_plot, upper_plot, color="orange", alpha=0.5)
plt.ylabel("Touch")
plt.xlabel("Timesteps")
pp = PdfPages(os.path.join(out_path, "p.pdf"))
pp.savefig()
pp.close()
plt.close()

plt.figure(figsize=(6, 2))
plt.plot(x_test[:100, 18], c="k")
plt.plot(y_prev_mu[:100, 18], c="b")
plt.plot(y_vis_mu[:100, 18], c="orange", linestyle="dashed")
lower_plot = y_vis_mu[:100, 18] - y_vis_std[:100, 18]
upper_plot = y_vis_mu[:100, 18] + y_vis_std[:100, 18]
plt.fill_between(np.arange(100), lower_plot, upper_plot, color="orange", alpha=0.5)
plt.ylabel("Sound")
plt.xlabel("Timesteps")
pp = PdfPages(os.path.join(out_path, "s.pdf"))
pp.savefig()
pp.close()
plt.close()
