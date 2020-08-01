"""Dataset definitions."""
import os
import torch
import torchvision.transforms as transforms
import numpy as np


class UR10Dataset(torch.utils.data.Dataset):
    """UR10 dataset containing move and grasp actions."""

    def __init__(self, path, modality, action, mode):
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
        """
        self.path = path
        self.modality = modality
        self.action = action
        self.mode = mode
        self.num_modality = len(modality)

        self.data = []
        for m in modality:
            temp = []
            for a in action:
                temp.append(torch.load(os.path.join(path, "%s_%s_%s.pt" % (a, mode, m))))
            temp = torch.cat(temp, dim=0)
            self.data.append(temp)

        if len(action) < 2:
            self.ranges = np.load(os.path.join(path, "%s_%s_ranges.npy" % (action[0], mode)))
        else:
            ranges1 = np.load(os.path.join(path, "%s_%s_ranges.npy" % (action[0], mode)))
            ranges2 = np.load(os.path.join(path, "%s_%s_ranges.npy" % (action[1], mode)))
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

    def __getitem__(self, idx):
        sample = []
        for i in range(self.num_modality):
            x = self.data[i][idx].clone()
            if len(x.shape) == 3:
                x = (x.float().div(255) - 0.5) * 2
            else:
                x = x.div(3)
            sample.append(x)
        return sample

    def __len__(self):
        return len(self.data[0])

    def get_trajectory(self, idx):
        if idx >= len(self.ranges):
            print("You should give index less than %d" % len(self.ranges))
            assert idx < len(self.ranges)
        sample = []
        begin, end = self.ranges[idx]
        for i in range(self.num_modality):
            x = self.data[i][begin:end].clone()
            if len(x.shape) == 4:
                x = (x.float().div(255) - 0.5) * 2
            else:
                x = x.div(3)
            sample.append(x)
        return sample
