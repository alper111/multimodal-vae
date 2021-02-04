import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np


def txt_to_tensor(filename):
    file = open(filename, "r")
    lines = file.readlines()
    lines = [x.rstrip().split(" ") for x in lines]
    return torch.tensor(np.array(lines, dtype=np.float), dtype=torch.float)


def return_split(i, splits):
    if i < train_split_idx:
        return splits[0]
    elif i < val_split_idx:
        return splits[1]
    else:
        return splits[2]


if not os.path.exists("data"):
    os.makedirs("data")

data_path = sys.argv[1]
actions = ["grasp", "move"]
action_iters = [0, 0]
modality = ["img", "joint", "endpoint", "object"]  # first modality is always image.
splits = ["train", "val", "test"]
N = 50
train_split_idx = 40
val_split_idx = 45
data_dict = {}

for it, action in enumerate(actions):
    data_dict[action] = {}
    for s in splits:
        data_dict[action][s] = {}
        for mod in modality:
            data_dict[action][s][mod] = []
        data_dict[action][s]["range"] = []

    for i in range(N):
        if (i == train_split_idx) or (i == val_split_idx):
            for j, _ in enumerate(action_iters):
                action_iters[j] = 0

        print("%s - %d" % (action, i))
        action_path = os.path.join(data_path, "%d/%s" % (i, action))
        imgs = os.listdir(action_path)
        for mod in modality[1:]:
            imgs.remove("%s_%d.txt" % (mod, i))
        if ".DS_Store" in imgs:
            imgs.remove(".DS_Store")

        # save images
        tensor_1 = torch.tensor(plt.imread(os.path.join(action_path, "0.jpeg")), dtype=torch.uint8)
        for j in range(1, len(imgs)-1):
            tensor_2 = torch.tensor(plt.imread(os.path.join(action_path, "%d.jpeg" % j)), dtype=torch.uint8)
            tensor_cat = torch.cat([tensor_1, tensor_2], dim=2)
            split = return_split(i, splits)
            data_dict[action][split][modality[0]].append(tensor_cat)
            tensor_1 = tensor_2.clone()

        # save other modalities
        for mod in modality[1:]:
            x = txt_to_tensor(os.path.join(action_path, "%s_%d.txt" % (mod, i)))
            x_cat = torch.cat([x[:-1], x[1:]], dim=1)
            split = return_split(i, splits)
            data_dict[action][split][mod].append(x_cat)
        data_dict[action][split]["range"].append([action_iters[it], action_iters[it]+x_cat.shape[0]])
        action_iters[it] += x_cat.shape[0]

for action in actions:
    for s in splits:
        for mod in modality:
            if mod == modality[0]:
                data_dict[action][s][mod] = torch.stack(data_dict[action][s][mod], dim=0).permute(0, 3, 1, 2).contiguous()
            else:
                data_dict[action][s][mod] = torch.cat(data_dict[action][s][mod], dim=0)
            torch.save(data_dict[action][s][mod], "data/%s_%s_%s.pt" % (action, s, mod))
        np.save("data/%s_%s_range.npy" % (action, s), data_dict[action][s]["range"])
