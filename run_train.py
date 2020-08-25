import os
import yaml

sizes = list(range(1, 41))
N = 5
for i in range(N):
    for s in sizes:
        file = open(os.path.join("opts-trainjob.yml"), "w")
        opts = {}
        opts["save"] = "save/imgjoint-%d-%d" % (s, i)
        opts["device"] = "cuda"
        opts["batch_size"] = 32
        opts["epoch"] = 100
        opts["lambda"] = 1.0
        opts["beta"] = 0.0
        opts["init_method"] = "xavier"
        opts["lr"] = 0.001
        opts["in_blocks"] = [
            [-2, 1024, 128, 6, 32, 64, 64, 128, 128, 256],
            [-1, 14, 32, 64, 64, 128, 128, 256, 128]
        ]
        opts["in_shared"] = [256, 256]
        opts["out_shared"] = [128, 256]
        opts["out_blocks"] = [
            [-2, 128, 1024, 256, 256, 128, 128, 64, 64, 32],
            [-1, 128, 256, 128, 128, 64, 64, 32, 28]
        ]
        opts["traj_count"] = s
        yaml.dump(opts, file)
        file.close()
        print("Started training with %d trajectories, #%d" % (s, i))
        os.system("python train.py -opts opts-trainjob.yml")
        os.system("python test.py -opts save/imgjoint-%d-%d/opts.yaml -banned 0 0 -prefix both" % (s, i))
        os.system("python test.py -opts save/imgjoint-%d-%d/opts.yaml -banned 0 1 -prefix img" % (s, i))
        os.system("python test.py -opts save/imgjoint-%d-%d/opts.yaml -banned 1 0 -prefix joint" % (s, i))
