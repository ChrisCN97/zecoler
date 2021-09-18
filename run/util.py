import os
import shutil
from server import S1, S2, USER, IP
import matplotlib.pyplot as plt
import numpy as np

def get_clear_folder(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
    return folder

def scp_get(source, target, port, user=USER, ip=IP):
    os.system("scp -r -P {} {}@{}:{} {}".format(port, user, ip, source, target))

def get_dataset(method, task, lang, size, from_server):
    source = os.path.join(from_server["root"], "method", method, "dataset", task, lang, str(size))
    target = os.path.join("../method", method, "dataset", task, lang)
    if not os.path.exists(target):
        os.mkdir(target)
    target = os.path.join(target, str(size))
    scp_get(source, target, from_server["port"])

def get_output(method, task, name, from_server):
    source = os.path.join(from_server["root"], "run/output", task, method, name)
    target = os.path.join("output", task, method)
    scp_get(source, target, from_server["port"])

def plot_loss(folder, name):
    # name: acc.npy / loss.npy
    loss_list = np.load(os.path.join(folder, name))
    print(len(loss_list))
    plt.figure()
    plt.plot(np.arange(len(loss_list)), loss_list)
    plt.title("{}: {}".format(folder, name))
    plt.show()

if __name__ == "__main__":
    # get_output(method="ptuning", task="clone_detection", name="Java_5000", from_server=S1)
    get_dataset(method="ptuning", task="clone_detection", lang="C", size="32", from_server=S2)
    # plot_loss(folder="output/clone_detection/ptuning/Java_5000_2/p10-i0", name="acc.npy")