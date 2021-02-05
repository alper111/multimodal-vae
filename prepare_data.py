import os
import argparse

import torch
import yaml
import matplotlib.pyplot as plt
import numpy as np

import utils


parser = argparse.ArgumentParser("Prepare dataset.")
parser.add_argument("-opts", help="Dataset options file", type=str, required=True)
args = parser.parse_args()

opts = yaml.safe_load(open(args.opts, "r"))

if not os.path.exists("data"):
    os.makedirs("data")

action_iters = [0 for _ in opts["actions"]]
splits = ["train", "val", "test"]
data_dict = {}
order = list(range(opts["N"]))
if opts["shuffle"]:
    np.random.shuffle(order)

for it, action in enumerate(opts["actions"]):
    data_dict[action] = {}
    for s in splits:
        data_dict[action][s] = {}
        for mod in opts["modality"]:
            data_dict[action][s][mod] = []
        data_dict[action][s]["range"] = []

    for i, idx in enumerate(order):
        if (i == opts["sp_tr"]) or (i == opts["sp_vl"]):
            for j, _ in enumerate(action_iters):
                action_iters[j] = 0

        print("%s - %d" % (action, i))
        action_path = os.path.join(opts["path"], "%d/%s" % (idx, action))
        imgs = os.listdir(action_path)
        for mod in opts["modality"][1:]:
            imgs.remove("%s_%d.txt" % (mod, idx))
        if ".DS_Store" in imgs:
            imgs.remove(".DS_Store")

        # save images
        img_t = torch.tensor(plt.imread(os.path.join(action_path, "0.jpeg")), dtype=torch.uint8)
        for j in range(1, len(imgs)):
            img_tnext = torch.tensor(plt.imread(os.path.join(action_path, "%d.jpeg" % j)), dtype=torch.uint8)
            tensor_cat = torch.cat([img_t, img_tnext], dim=2)
            split = utils.return_split(i, splits, opts["sp_tr"], opts["sp_vl"])
            data_dict[action][split][opts["modality"][0]].append(tensor_cat)
            img_t = img_tnext.clone()

        # save other modalities
        for mod in opts["modality"][1:]:
            x = utils.txt_to_tensor(os.path.join(action_path, "%s_%d.txt" % (mod, idx)))
            x_cat = torch.cat([x[:-1], x[1:]], dim=1)
            split = utils.return_split(i, splits, opts["sp_tr"], opts["sp_vl"])
            data_dict[action][split][mod].append(x_cat)
        data_dict[action][split]["range"].append([action_iters[it], action_iters[it]+x_cat.shape[0]])
        action_iters[it] += x_cat.shape[0]

for action in opts["actions"]:
    for s in splits:
        for mod in opts["modality"]:
            if mod == opts["modality"][0]:
                data_dict[action][s][mod] = torch.stack(data_dict[action][s][mod], dim=0)
                data_dict[action][s][mod] = data_dict[action][s][mod].permute(0, 3, 1, 2).contiguous()
            else:
                data_dict[action][s][mod] = torch.cat(data_dict[action][s][mod], dim=0)
            torch.save(data_dict[action][s][mod], "data/%s_%s_%s.pt" % (action, s, mod))
        np.save("data/%s_%s_range.npy" % (action, s), data_dict[action][s]["range"])
