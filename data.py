"""Dataset definitions."""
import os

import torch
import numpy as np


class MyDataset(torch.utils.data.Dataset):
    """
    Dataset class for processed data.

    You should first pre-process data and save it to pytorch objects.
    See README and `prepare_data.py` for details.
    """

    def __init__(self, path, modality, action, mode, traj_list=None):
        """
        Initialize dataset with given options.

        Parameters
        ----------
        path : str
            Folder which contains data in pytorch objects.
        modality : list of str
            List of modalities to include in the data.
        action : list of str
            List of actions to include in the data.
        mode : str
            ["train" | "validation" | "test"]
        traj_list : list of int, optional
            List of trajectory ids to use.
        """
        self.path = path
        self.modality = modality
        self.action = action
        self.mode = mode
        self.num_modality = len(modality)
        self.traj_list = traj_list

        if len(action) < 2:
            self.ranges = np.load(os.path.join(path, "%s_%s_range.npy" % (action[0], mode)))
        else:
            ranges1 = np.load(os.path.join(path, "%s_%s_range.npy" % (action[0], mode)))
            ranges2 = np.load(os.path.join(path, "%s_%s_range.npy" % (action[1], mode)))
            self.ranges = []
            it = 0
            for start, end in ranges1:
                duration = end - start
                self.ranges.append([it, it+duration])
                it += duration
            for start, end in ranges2:
                duration = end - start
                self.ranges.append([it, it+duration])
                it += duration

        self.data = []
        for m in modality:
            temp = []
            for a in action:
                temp.append(torch.load(os.path.join(path, "%s_%s_%s.pt" % (a, mode, m))))
            temp = torch.cat(temp, dim=0)

            if traj_list is not None:
                filtered = []
                for t in traj_list:
                    begin, end = self.ranges[t]
                    filtered.append(temp[begin:end])
                filtered = torch.cat(filtered, dim=0)
                self.data.append(filtered)
            else:
                self.data.append(temp)

        self.scale = []
        self.offset = []
        for i in range(self.num_modality):
            xmodality = self.data[i]
            xmin = xmodality.min()
            xmax = xmodality.max()
            self.offset.append(xmin)
            self.scale.append(xmax-xmin)

    def __getitem__(self, idx):
        sample = [self.data[i][idx] for i in range(self.num_modality)]
        sample = self.normalize(sample)
        return sample

    def normalize(self, x):
        x_normed = []
        for i, x_i in enumerate(x):
            x_n = ((x_i.clone().float() - self.offset[i]) / self.scale[i]) * 2 - 1
            x_normed.append(x_n)
        return x_normed

    def denormalize(self, x):
        x_denormed = []
        for i, x_i in enumerate(x):
            x_n = (x_i.clone()*0.5+0.5) * self.scale[i] + self.offset[i]
            x_denormed.append(x_n)
        return x_denormed

    def __len__(self):
        return len(self.data[0])

    def get_trajectory(self, idx):
        if idx >= len(self.ranges):
            print("You should give index less than %d" % len(self.ranges))
            assert idx < len(self.ranges)

        if self.traj_list is not None:
            print("This method is not compatible when data is initialized with traj_list option.")
            assert self.traj_list is None

        sample = []
        begin, end = self.ranges[idx]
        sample = [self.data[i][begin:end] for i in range(self.num_modality)]
        sample = self.normalize(sample)
        return sample
