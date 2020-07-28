"""Train multimodal VAE."""
import os
import time
import argparse
import yaml
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import models

parser = argparse.ArgumentParser("Train effect prediction models.")
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

# x_train = torch.tensor(np.load("data/train.npy"), dtype=torch.float, device=opts["device"])
# x_test = torch.tensor(np.load("data/test.npy"), dtype=torch.float, device=opts["device"])
# LOAD DATA
train_grasp_img = torch.load("data/grasp_train_img.pt").float() / 255.0
train_grasp_joint = torch.load("data/grasp_train_joint.pt")
train_move_img = torch.load("data/move_train_img.pt").float() / 255.0
train_move_joint = torch.load("data/move_train_joint.pt")
test_grasp_img = torch.load("data/grasp_test_img.pt").float() / 255.0
test_grasp_joint = torch.load("data/grasp_test_joint.pt")
test_move_img = torch.load("data/move_test_img.pt").float() / 255.0
test_move_joint = torch.load("data/move_test_joint.pt")

x_train_img = torch.cat([train_grasp_img, train_move_img], dim=0)
x_train_joint = torch.cat([train_grasp_joint, train_move_joint], dim=0)
x_test_img = torch.cat([test_grasp_img, test_move_img], dim=0)
x_test_joint = torch.cat([test_grasp_joint, test_move_joint], dim=0)

idx = int(0.8 * x_train_img.shape[0])
x_train_img, x_val_img = x_train_img[:idx], x_train_img[idx:]
x_train_joint, x_val_joint = x_train_joint[:idx], x_train_joint[idx:]

# train_set = torch.utils.data.TensorDataset(x_train)
# loader = torch.utils.data.DataLoader(train_set, batch_size=opts["batch_size"], shuffle=True)

model = models.MultiVAE(
    in_blocks=opts["in_blocks"],
    in_shared=opts["in_shared"],
    out_shared=opts["out_shared"],
    out_blocks=opts["out_blocks"],
    init=opts["init_method"])
model.to(opts["device"])
optimizer = torch.optim.Adam(lr=opts["lr"], params=model.parameters(), amsgrad=True)
print(model)

for e in range(opts["epoch"]):
    running_avg = 0.0
    R = torch.randperm(idx)
    for i in tqdm(range(idx // opts["batch_size"])):
        optimizer.zero_grad()

        x_i_img = x_train_img[R[i*opts["batch_size"]:(i+1)*opts["batch_size"]]].to(opts["device"])
        x_i_joint = x_train_joint[R[i*opts["batch_size"]:(i+1)*opts["batch_size"]]].to(opts["device"])

        y_i_img = x_i_img.clone()
        y_i_joint = x_i_joint.clone()

        if np.random.rand() < 0.5:
            x_i_img[:, 3:] = 0.0
            x_i_joint[:, 6:] = -5.0

        x_all = [x_i_img, x_i_joint]
        y_all = [y_i_img, y_i_joint]

        loss = model.loss(x_all, y_all, lambd=opts["lambda"], beta=opts["beta"])
        loss.backward()
        optimizer.step()
        running_avg += loss.item()
    running_avg /= (i+1)
    with torch.no_grad():
        y_val_img = x_val_img.clone()
        y_val_joint = x_val_joint.clone()
        if np.random.rand() < 0.5:
            x_val_img[:, 3:] = 0.0
            x_val_joint[:, 6:] = -5.0
        x_val_all = [x_val_img, x_val_joint]
        y_val_all = [y_val_img, y_val_joint]
        mse_val = model.mse_loss(x_val_all, y_val_all, sample=False, beta=0.0)

    writer.add_scalar("Epoch loss", running_avg, e)
    writer.add_scalar("MSE val",  mse_val, e)
    print("Epoch loss: %.3f" % running_avg)
    print("MSE validation: %.3f" % mse_val)

    model.save(opts["save"], "multivae_%d" % e)
    model.save(opts["save"], "multivae_last")
