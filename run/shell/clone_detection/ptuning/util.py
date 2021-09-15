import matplotlib.pyplot as plt
import numpy as np

def plot_loss(loss_list_name, title):
    loss_list = np.load(loss_list_name)
    print(len(loss_list))
    plt.figure()
    plt.plot(np.arange(len(loss_list)), loss_list)
    plt.title(loss_list_name + ": " + title)
    plt.show()

if __name__ == '__main__':
    pass