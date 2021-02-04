# Multimodal VAE - PyTorch

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/5a7dc413c50e47b58350982f1c9d3d07)](https://app.codacy.com/gh/alper111/multimodal-vae?utm_source=github.com&utm_medium=referral&utm_content=alper111/multimodal-vae&utm_campaign=Badge_Grade)

This repository contains PyTorch implementation of the paper "Multimodal representation models for prediction and control from partial information".

## Download data and process
```bash
python get_drive_file.py 1Nn-ONccUbW1cBwm6nRhgF-zjtKoB8zO6 data2020.zip
unzip data2020.zip
python prepare_data.py data2020
rm -r data2020
rm data2020.zip
```

### Data folder structure
```
/mydataset
    /1 (trajectory)
        /action1 (action label)
            0.jpeg (only jpeg for now)
            1.jpeg
            ...
            N.jpeg (N can be any number)
            objects.txt [N x D] matrix
        /action2 (optional)
    /2
    ...
    /M
```


## Example opts.yaml
```yaml
save: save/test1
device: cuda
batch_size: 32
epoch: 100
lambda: 1.0
beta: 0.0
init_method: xavier
lr: 0.001
reduce: true
mse: true
beta_decay: 0.0
in_blocks: [
  [-2, 1024, 128, 6, 32, 64, 64, 128, 128, 256],
  [-1, 14, 32, 64, 64, 128, 128, 256, 128]
]
in_shared: [256, 256]
out_shared: [128, 256]
out_blocks: [
  [-2, 128, 1024, 256, 256, 128, 128, 64, 64, 32],
  [-1, 128, 256, 128, 128, 64, 64, 32, 28]
]
traj_count: 40
```

## Train the model
```bash
python train.py -opts opts.yaml -mod img
```

You can watch the training progress with tensorboard:
```bash
tensorboard --logdir <savefolder>/log
```