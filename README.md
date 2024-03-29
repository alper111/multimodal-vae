# Multimodal VAE - PyTorch

[![Codacy Badge](https://app.codacy.com/project/badge/Grade/bc54bed0d2404f94977bc3f92bee5521)](https://www.codacy.com/gh/alper111/multimodal-vae/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=alper111/multimodal-vae&amp;utm_campaign=Badge_Grade)

This repository contains PyTorch implementation of the paper "Multimodal representation models for prediction and control from partial information".

## Download data and process
```bash
python get_drive_file.py 1Nn-ONccUbW1cBwm6nRhgF-zjtKoB8zO6 data2020.zip
unzip data2020.zip
# see data.yml definition below
python prepare_data.py -opts data.yml
rm -r data2020
rm data2020.zip
```

### Data folder structure
```
/mydataset
    /0 (trajectory)
        /action1 (action label)
            0.jpeg (only jpeg for now)
            1.jpeg
            ...
            N.jpeg (N can be any number)
            objects_0.txt [N x D] matrix
            anything_0.txt [N x K] matrix
        /action2 (optional)
    /1
        /action1
            0.jpeg
            1.jpeg
            ...
            P.jpeg (can be different from N above)
            objects_1.txt [P x D] matrix
            anything_1.txt [P x K] matrix
    ...
    /M
```

In order to pre-process the data and run the training script, you will need two yaml option files. Here are examples:

## Example data.yml
```yaml
path: "mydataset"
actions: ["action1"]
modality: ["img", "anything", "objects"]  # first modality should be always img
N: 10
sp_tr: 7  # train split
sp_vl: 8  # validation split
shuffle: false  # whether to shuffle trajectories
```

## Example opts.yml
```yaml
save: save/test
data: data_folder_path
device: cuda
modality: ["img", "anything", "objects"]
action: ["action1"]
batch_size: 128
epoch: 10
lambda: 1.0
beta: 0.0
init_method: xavier
lr: 0.0005
reduce: true
mse: true
beta_decay: 0.0
in_blocks: [
  [-2, 1024, 128, 6, 32, 64, 64, 128, 128, 256],  # image encoder
  [-1, 28, 32, 64, 64, 128, 128, 256, 128],  # anything encoder
  [-1, 16, 32, 64, 64, 128, 128, 256, 128]  # objects encoder
]
in_shared: [384, 256]  # shared encoder
out_shared: [128, 384]  # shared decoder
out_blocks: [
  [-2, 128, 1024, 256, 256, 128, 128, 64, 64, 32],  # image decoder
  [-1, 128, 256, 128, 128, 64, 64, 32, 56],  # anything decoder
  [-1, 128, 256, 128, 128, 64, 64, 32, 32]  # objects decoder
]
traj_count: 6
```

## Prepare the dataset
```bash
python prepare_data.py -opts data.yaml
```

## Train the model
```bash
python train.py -opts opts.yaml
```

You can watch the training progress with tensorboard:
```bash
tensorboard --logdir <savefolder>/log
```

## Test the model
While testing, you can optionally ban some modalities to test accurate the model reconstructs and forecasts previous and next timesteps.

Without ban:
```bash
python test.py -opts opts.yml -banned 0 0 0 -prefix no_ban
```

Or, ban the `objects` modality:
```bash
python test.py -opts opts.yml -banned 0 0 1 -prefix ban_objects
```
