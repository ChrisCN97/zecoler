import os.path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 24
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['axes.linewidth'] = 3  # bounding box line width
file_path = "pics"


def zero_shot_plot(data_set, filename, xlabel, ylabel='Accuracy', legend_list=None):
    fig, axes = plt.subplots(1, 1, figsize=(10, 8))
    colors = ['c', 'b', 'y', 'm']
    markers = ['o', 's', 'd', '^', '+']

    for idx, (x, y) in enumerate(data_set):
        if legend_list is None:
            plt.plot(x, y, colors[idx], marker=markers[idx], markersize=15, linewidth=5)
        else:
            plt.plot(x, y, colors[idx], marker=markers[idx], markersize=15, linewidth=5, label=legend_list[idx])
    if legend_list is not None:
        plt.legend()
    plt.grid()
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=24)
    plt.ylabel(ylabel, fontsize=24)

    plt.show()
    fig.tight_layout()
    save_fig(fig, filename)

def save_fig(fig, filename):
    plt.show()
    file = os.path.join(file_path, filename)
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    fig.savefig(file, dpi=200)

def plot_fig3():
    x = [1,
         5,
         10,
         15,
         20
         ]

    y = [83.5,
         87.8,
         88.9,
         85.2,
         87.7]

    for i in range(len(y)):
        y[i] /= 100

    xlabel = '# Number of Prompts'
    file = 'fig3.png'

    zero_shot_plot(data_set=[(x, y)], filename=file, xlabel=None)


def plot_fig4():
    x = ['head',
         'middle',
         'uniformly',
         'tail'
         ]

    y = [81.4,
         86.4,
         88.9,
         81.3]

    for i in range(len(y)):
        y[i] /= 100

    xlabel = '# Position of Prompts'
    file = 'fig4.png'

    zero_shot_plot(data_set=[(x, y)], filename=file, xlabel=None)

def plot_tab3_single(x, y_list, legend_list, filename):
    data_set = []
    for y in y_list:
        for i in range(len(y)):
            y[i] /= 100
        data_set.append((x, y))
    xlabel = 'Data Size'
    zero_shot_plot(data_set=data_set, filename=filename, xlabel=xlabel, legend_list=legend_list)

def plot_tab3_CD():
    x = [32, 100, 300, 500, 700]
    legend_list = ["Zecoler", "CodeBERT", "CodeBERTa"]
    # zecoler, CodeBERT, CodeBERTa
    Java_list = [
        [53.3, 63.6, 85.8, 90.8, 95.1],
        [48.2, 55.4, 51.3, 48.8, 48.1],
        [52.8, 51, 53.9, 50.7, 51.6]
    ]
    SC_list = [
        [90.1, 93.9, 94.3, 93.6, 94.4],
        [65.4, 68.7, 69.4, 70.2, 75.3],
        [65.1, 64.5, 65.0, 68.3, 73.7]
    ]
    Go_list = [
        [52.8, 99.5, 99.3, 99.1, 99.4],
        [57.9, 50.3, 49.5, 71.2, 68.4],
        [57.9, 53.1, 52, 65.3, 65.9]
    ]

    plot_tab3_single(x, Java_list, legend_list, filename="cd_java_fs.png")
    plot_tab3_single(x, SC_list, legend_list, filename="cd_sc_fs.png")
    plot_tab3_single(x, Go_list, legend_list, filename="cd_go_fs.png")

def plot_tab3_CS():
    x = [32, 100, 300, 500, 700]
    legend_list = ["Zecoler", "CodeBERT", "CodeBERTa"]
    # zecoler, CodeBERT, CodeBERTa
    Java_list = [
        [51.7, 52.6, 51.7, 57.5, 61.9],
        [51.7, 53.6, 50.8, 51.7, 50.3],
        [55.1, 50.4, 52.8, 52, 50.4]
    ]
    SC_list = [
        [53.2, 63, 90.1, 91.7, 92.1],
        [52.4, 52.6, 56.5, 53.2, 52.2],
        [53, 54.9, 54.3, 58.8, 55.8]
    ]
    Go_list = [
        [52, 95.7, 99.5, 99.3, 99.4],
        [53.2, 47.9, 49.5, 44.8, 68.4],
        [50.9, 48.9, 53.9, 45, 45.8]
    ]

    plot_tab3_single(x, Java_list, legend_list, filename="cs_java_fs.png")
    plot_tab3_single(x, SC_list, legend_list, filename="cs_sc_fs.png")
    plot_tab3_single(x, Go_list, legend_list, filename="cs_go_fs.png")

def plot_tab3_MNP():
    x = [32, 100, 300, 500, 700]
    legend_list = ["Zecoler", "CodeBERT", "CodeBERTa"]
    # zecoler, CodeBERT, CodeBERTa
    Java_list = [
        [66.9, 72.8, 98.7, 99.8, 99.8],
        [52.5, 50.2, 49.3, 50.3, 50.4],
        [49.4, 52, 52.9, 50.8, 50.2]
    ]
    SC_list = [
        [52.7, 62.8, 88.8, 90.3, 93.1],
        [52.1, 49.6, 53.9, 53.7, 51.7],
        [50.2, 52.2, 53.9, 61.8, 64.6]
    ]
    Go_list = [
        [52.2, 77.7, 99.2, 99.3, 99.4],
        [49.7, 49.5, 49.5, 47.6, 68.4],
        [49.2, 48.7, 50.9, 47.4, 48.6]
    ]

    plot_tab3_single(x, Java_list, legend_list, filename="mnp_java_fs.png")
    plot_tab3_single(x, SC_list, legend_list, filename="mnp_sc_fs.png")
    plot_tab3_single(x, Go_list, legend_list, filename="mnp_go_fs.png")

def plot_fig5():
    # plt.style.use('ggplot')

    labels = ["Java", "Python", "Go", "Solidity", "Haskell", "Kotlin", "PHP", "C#", "Fortran"]
    x = np.arange(len(labels))  # the label locations
    margin = 0.1

    y_labels = ["Java", "Go", "Python"]
    y_color = ["#FAE3D9", "#BBDED6", "#8AC6D1"]
    width = (1. - 2. * margin) / len(y_labels)  # the width of the bars
    y_res = [
        [98.8, 91.6, 96.4, 79.8, 91.2, 96.8, 96.3, 96.6, 92.7],
        [95.5, 91.8, 99.5, 69.9, 90.6, 94.4, 95.5, 94.3, 91.2],
        [94.1, 98.5, 94.1, 53.6, 83.4, 96.4, 93.6, 91.9, 81.5]
    ]
    for y in y_res:
        for i in range(len(y)):
            y[i] /= 100

    fig, ax = plt.subplots(figsize=(10, 5))
    # plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
    plt.grid(zorder=0)
    for idx, y in enumerate(y_res):
        ax.bar(x + idx*width, y, width, label=y_labels[idx], zorder=50, color=y_color[idx])

    fontsize = 15
    ax.set_ylim([0.4, 1.05])
    ax.set_yticks(np.arange(0.4, 1.2, step=0.2))
    ax.set_xlabel('Program Languages', fontsize=fontsize)
    ax.set_ylabel('Accuracy', fontsize=fontsize)
    ax.set_xticks(x+width)
    ax.set_xticklabels(labels)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    ax.spines['bottom'].set(zorder=200)

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
               ncol=3, mode="expand", borderaxespad=0., fontsize=fontsize)


    fig.tight_layout()

    save_fig(fig, filename="fig5.png")

def plot_rq2_CD():
    x = [0, 32, 100, 300, 500, 700]
    legend_list = ["Zecoler", "CodeBERT"]

    SC_list = [
        [79.8, 95.1, 95, 94.7, 93.9, 94.3],
        [65.3, 75.1, 77.7, 81.9, 82.0, 83.0]
    ]
    Go_list = [
        [96.4, 97.1, 97.7, 99.1, 98.1, 99],
        [91.7, 89.2, 49.8, 50.3, 93.7, 94.1]
    ]

    plot_tab3_single(x, SC_list, legend_list, filename="cd_sc_rq2.png")
    plot_tab3_single(x, Go_list, legend_list, filename="cd_go_rq2.png")

def plot_rq2_CS():
    x = [0, 32, 100, 300, 500, 700]
    legend_list = ["Zecoler", "CodeBERT"]

    SC_list = [
        [67.1, 51, 75.8, 88.5, 91.1, 93.1],
        [48.9, 49.9, 52.5, 55.8, 53.5, 53]
    ]
    Go_list = [
        [80.3, 62.6, 51.6, 98.7, 99.6, 99.4],
        [46.2, 51.4, 50.8, 49.5, 74.2, 65]
    ]

    plot_tab3_single(x, SC_list, legend_list, filename="cs_sc_rq2.png")
    plot_tab3_single(x, Go_list, legend_list, filename="cs_go_rq2.png")

def plot_rq2_MNP():
    x = [0, 32, 100, 300, 500, 700]
    legend_list = ["Zecoler", "CodeBERT"]

    SC_list = [
        [59.2, 56.4, 76.6, 87.7, 91.7, 91.8],
        [52.1, 52.6, 63.5, 53.7, 58.7, 66.1]
    ]
    Go_list = [
        [98.8, 56.7, 63.3, 98.9, 99.4, 99.3],
        [65.2, 57.4, 64.6, 63.4, 74.8, 74.4]
    ]

    plot_tab3_single(x, SC_list, legend_list, filename="mnp_sc_rq2.png")
    plot_tab3_single(x, Go_list, legend_list, filename="mnp_go_rq2.png")

if __name__ == '__main__':
    # plot_tab3_CD()
    # plot_tab3_CS()
    # plot_tab3_MNP()
    # plot_fig3()
    plot_fig4()
    # plot_fig3()
    # plot_rq2_CD()
    # plot_rq2_CS()
    # plot_rq2_MNP()
