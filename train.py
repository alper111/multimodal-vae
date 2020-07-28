"""Train multimodal VAE."""
import os
import sys
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import models

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

if len(sys.argv) < 2:
    print("Usage: python train.py <run_name>")
    exit()
run_name = sys.argv[1]
logdir = os.path.join("runs", run_name)
if os.path.exists(logdir):
    os.system("rm -rf %s" % logdir)
writer = SummaryWriter(logdir)

x_train = torch.tensor(np.load("data/train.npy"), dtype=torch.float, device=device)
x_test = torch.tensor(np.load("data/test.npy"), dtype=torch.float, device=device)
idx = int(0.8 * x_train.shape[0])
x_train, x_val = x_train[:idx], x_train[idx:]

train_set = torch.utils.data.TensorDataset(x_train)
loader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=True)
epoch = 2000

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
optimizer = torch.optim.Adam(lr=1e-4, params=model.parameters(), amsgrad=True)
print(model)

for e in range(epoch):
    running_avg = 0.0
    for i, (x,) in enumerate(tqdm(loader)):
        optimizer.zero_grad()
        loss = model.loss(x[:, :28], x[:, 28:])
        loss.backward()
        optimizer.step()
        running_avg += loss.item()
    running_avg /= (i+1)
    with torch.no_grad():
        mse_train = model.mse_loss(x_train[:, :28], x_train[:, 28:], sample=False, beta=0.0)
        mse_val = model.mse_loss(x_val[:, :28], x_val[:, 28:], sample=False, beta=0.0)

    writer.add_scalar("Epoch loss", running_avg, e)
    writer.add_scalars("MSE train/val", {"train": mse_train, "val": mse_val}, e)
    print("Epoch loss: %.3f" % running_avg)
    print("MSE train: %.3f" % mse_train)
    print("MSE validation: %.3f" % mse_val)

model.save(os.path.join("save", run_name), "multivae")
