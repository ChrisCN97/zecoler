import matplotlib.pyplot as plt
import numpy as np
import os

def plot_loss(output, folder, name):
    # name: acc.npy / loss.npy
    loss_list = np.load(os.path.join("../../../output/clone_detection/ptuning", output, folder, name))
    print(len(loss_list))
    plt.figure()
    plt.plot(np.arange(len(loss_list)), loss_list)
    plt.title("{}: {}".format(output, name))
    plt.show()

if __name__ == '__main__':
    plot_loss("Java_5000", "p10-i0", "acc.npy")