import numpy as np
from matplotlib import pyplot as plt

def plot(top1a, top1e, top5a, top5e):
    plt.figure(0)


    y = np.arange(len(top1a))

    plt.subplot(1,2,1)
    plt.title("Top 1 error rate")
    plt.xlabel("Epoch")
    plt.ylim(0, 0.3)
    plt.plot(y, top1e)
    plt.axhline(y = top1e[0], color = 'r', linestyle = '-')

    plt.subplot(1,2,2)
    plt.xlabel("Epoch")
    plt.title("Top 5 error rate")
    plt.ylim(0, 0.3)
    plt.plot(y, top5e)
    plt.axhline(y = top5e[0], color = 'r', linestyle = '-')
    plt.savefig("plots/errorrates.png")

    plt.figure(1)

    plt.subplot(1,2,1)
    plt.ylim(0, 0.3)
    plt.title("Top 1 accuracy")
    plt.xlabel("Epoch")
    plt.plot(y, top1a)
    plt.axhline(y = top1a[0], color = 'r', linestyle = '-')

    plt.subplot(1,2,2)
    plt.title("Top 5 accuracy")
    plt.xlabel("Epoch")
    plt.ylim(0, 0.3)
    plt.plot(y, top5a)
    plt.axhline(y = top5a[0], color = 'r', linestyle = '-')
    plt.savefig("plots/accuracies.png")
