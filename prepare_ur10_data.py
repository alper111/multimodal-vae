import os
import torch
import matplotlib.pyplot as plt
import numpy as np


def txt_to_tensor(filename):
    file = open(filename, "r")
    lines = file.readlines()
    lines = [x.rstrip().split(" ") for x in lines]
    return torch.tensor(np.array(lines, dtype=np.float), dtype=torch.float)


grasp_train_img = []
grasp_train_joint = []
grasp_train_obj = []
grasp_test_img = []
grasp_test_joint = []
grasp_test_obj = []
move_train_img = []
move_train_joint = []
move_train_obj = []
move_test_img = []
move_test_joint = []
move_test_obj = []


for action in ["grasp", "move"]:
    for i in range(50):
        print("%s - %d" % (action, i))
        action_path = "data/ur10/%d/%s" % (i, action)
        imgs = os.listdir(action_path)
        imgs.remove("demonstration_%d.txt" % i)
        imgs.remove("object_%d.txt" % i)
        if ".DS_Store" in imgs:
            imgs.remove(".DS_Store")

        tensor_1 = torch.tensor(plt.imread(os.path.join(action_path, "0.jpeg")), dtype=torch.uint8)
        for j in range(1, len(imgs)-1):
            tensor_2 = torch.tensor(plt.imread(os.path.join(action_path, "%d.jpeg" % j)), dtype=torch.uint8)
            tensor_cat = torch.cat([tensor_1, tensor_2], dim=2)
            if i < 40:
                if action == "grasp":
                    grasp_train_img.append(tensor_cat)
                else:
                    move_train_img.append(tensor_cat)
            else:
                if action == "grasp":
                    grasp_test_img.append(tensor_cat)
                else:
                    move_test_img.append(tensor_cat)
            tensor_1 = tensor_2.clone()
        joints = txt_to_tensor(os.path.join(action_path, "demonstration_%d.txt" % i))
        obj_pos = txt_to_tensor(os.path.join(action_path, "object_%d.txt" % i))
        # [x[t-1], x[t]]
        joints_cat = torch.cat([joints[:-1], joints[1:]], dim=1)
        obj_cat = torch.cat([obj_pos[:-1], obj_pos[1:]], dim=1)
        if i < 40:
            if action == "grasp":
                grasp_train_joint.append(joints_cat)
                grasp_train_obj.append(obj_cat)
            else:
                move_train_joint.append(joints_cat)
                move_train_obj.append(obj_cat)
        else:
            if action == "grasp":
                grasp_test_joint.append(joints_cat)
                grasp_test_obj.append(obj_cat)
            else:
                move_test_joint.append(joints_cat)
                move_test_obj.append(obj_cat)

grasp_train_img = torch.stack(grasp_train_img, dim=0).permute(0, 3, 1, 2)
grasp_train_joint = torch.cat(grasp_train_joint, dim=0)
grasp_train_obj = torch.cat(grasp_train_obj, dim=0)
grasp_test_img = torch.stack(grasp_test_img, dim=0).permute(0, 3, 1, 2)
grasp_test_joint = torch.cat(grasp_test_joint, dim=0)
grasp_test_obj = torch.cat(grasp_test_obj, dim=0)
move_train_img = torch.stack(move_train_img, dim=0).permute(0, 3, 1, 2)
move_train_joint = torch.cat(move_train_joint, dim=0)
move_train_obj = torch.cat(move_train_obj, dim=0)
move_test_img = torch.stack(move_test_img, dim=0).permute(0, 3, 1, 2)
move_test_joint = torch.cat(move_test_joint, dim=0)
move_test_obj = torch.cat(move_test_obj, dim=0)

torch.save(grasp_train_img, "data/grasp_train_img.pt")
torch.save(grasp_train_joint, "data/grasp_train_joint.pt")
torch.save(grasp_train_obj, "data/grasp_train_obj.pt")
torch.save(grasp_test_img, "data/grasp_test_img.pt")
torch.save(grasp_test_joint, "data/grasp_test_joint.pt")
torch.save(grasp_test_obj, "data/grasp_test_obj.pt")
torch.save(move_train_img, "data/move_train_img.pt")
torch.save(move_train_joint, "data/move_train_joint.pt")
torch.save(move_train_obj, "data/move_train_obj.pt")
torch.save(move_test_img, "data/move_test_img.pt")
torch.save(move_test_joint, "data/move_test_joint.pt")
torch.save(move_test_obj, "data/move_test_obj.pt")
