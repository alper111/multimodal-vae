import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

N = 5
trajs = list(range(1, 41))
modality = ["both", "end", "joint"]
T = len(trajs)
yp_mean = np.zeros(T)
yp_std = np.zeros(T)
zp_mean = np.zeros(T)
zp_std = np.zeros(T)
yj_mean = np.zeros((T, 7))
yj_std = np.zeros((T, 7))
zj_mean = np.zeros((T, 7))
zj_std = np.zeros((T, 7))

for m in modality:
    print(m)
    for i, t in enumerate(trajs):
        ype = np.zeros(N)
        zpe = np.zeros(N)
        yje = np.zeros((N, 7))
        zje = np.zeros((N, 7))
        for j in range(N):
            file = open("save/imgjoint-%d-%d/outs/%s-result.txt" % (t, j, m), "r")
            lines = file.readlines()

            ype[j] = float(lines[10])
            zpe[j] = float(lines[11])
            yje[j] = np.array(lines[12].rstrip().split(","), dtype=np.float)
            zje[j] = np.array(lines[13].rstrip().split(","), dtype=np.float)

        yp_mean[i] = ype.mean()
        yp_std[i] = ype.std()
        zp_mean[i] = zpe.mean()
        zp_std[i] = zpe.std()
        yj_mean[i] = yje.mean(axis=0)
        yj_std[i] = yje.std(axis=0)
        zj_mean[i] = zje.mean(axis=0)
        zj_std[i] = zje.std(axis=0)
    print("one-step pixel")
    print(yp_mean, yp_std)
    print("forecast pixel")
    print(zp_mean, zp_std)
    print("one-step joint")
    for i in range(7):
        print(yj_mean[:, i], yj_std[:, i])
    print("forecast joint")
    for i in range(7):
        print(zj_mean[:, i], zj_std[:, i])

    plt.plot(trajs, yp_mean, color="b")
    plt.fill_between(trajs, yp_mean-yp_std, yp_mean+yp_std, color="b", alpha=0.2)
    pp = PdfPages("out/yp-%s.pdf" % m)
    pp.savefig()
    pp.close()
    plt.close()

    plt.plot(trajs, zp_mean, "b")
    plt.fill_between(trajs, zp_mean-zp_std, zp_mean+zp_std, color="b", alpha=0.2)
    pp = PdfPages("out/zp-%s.pdf" % m)
    pp.savefig()
    pp.close()
    plt.close()

    fig, ax = plt.subplots(3, 2, figsize=(12, 10))
    for i in range(3):
        for j in range(2):
            ax[i][j].plot(trajs, yj_mean[:, i*2+j], color="b")
            ax[i][j].fill_between(trajs, yj_mean[:, i*2+j]-yj_std[:, i*2+j], yj_mean[:, i*2+j]+yj_std[:, i*2+j], color="b", alpha=0.2)
    pp = PdfPages("out/yj-%s.pdf" % m)
    pp.savefig(fig)
    pp.close()
    plt.close()

    fig, ax = plt.subplots(3, 2, figsize=(12, 10))
    for i in range(3):
        for j in range(2):
            ax[i][j].plot(trajs, zj_mean[:, i*2+j], color="b")
            ax[i][j].fill_between(trajs, zj_mean[:, i*2+j]-zj_std[:, i*2+j], zj_mean[:, i*2+j]+zj_std[:, i*2+j], color="b", alpha=0.2)
    pp = PdfPages("out/zj-%s.pdf" % m)
    pp.savefig(fig)
    pp.close()
    plt.close()
