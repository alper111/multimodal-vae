import os

import numpy as np


for i in [1, 2, 4, 8, 10, 15, 20, 30, 40]:
    x1 = []
    x2 = []
    y1 = []
    y2 = []
    z1 = []
    z2 = []
    for j in range(5):
        path = "save/imgjoint-%s-%s" % (i, j)
        both = open(os.path.join(path, "outs/both-result.txt"), "r").readlines()
        img = open(os.path.join(path, "outs/img-result.txt"), "r").readlines()
        joint = open(os.path.join(path, "outs/joint-result.txt"), "r").readlines()
        x1.append(float(both[11]))
        x2.append(np.array(both[13].split(","), dtype=np.float))
        y1.append(float(img[11]))
        y2.append(np.array(img[13].split(","), dtype=np.float))
        z1.append(float(joint[11]))
        z2.append(np.array(joint[13].split(","), dtype=np.float))
    x2 = np.vstack(x2)
    y2 = np.vstack(y2)
    z2 = np.vstack(z2)
    print("="*80)
    print("%d TRAJECTORY" % i)
    print("img+joint -> img: %.5f +- %.5f" % (np.mean(x1), np.std(x1, ddof=1)))
    print("img -> img: %.5f +- %.5f" % (np.mean(y1), np.std(y1, ddof=1)))
    print("joint -> img: %.5f +- %.5f" % (np.mean(z1), np.std(z1, ddof=1)))
    print("")
    print("img+joint -> joint: %.5f +- %.5f" % (np.mean(x2), np.std(x2, ddof=1)))
    print("img -> joint: %.5f +- %.5f" % (np.mean(y2), np.std(y2, ddof=1)))
    print("joint -> joint: %.5f +- %.5f" % (np.mean(z2), np.std(z2, ddof=1)))
