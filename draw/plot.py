import os.path

import matplotlib as mpl
import matplotlib.pyplot as plt
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
    plt.xlabel(xlabel, fontsize=24)
    plt.ylabel(ylabel, fontsize=24)

    plt.show()
    fig.tight_layout()
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
         79.2,
         87.7]

    for i in range(len(y)):
        y[i] /= 100

    xlabel = '# Number of Prompts'
    file = 'zero_shot_fig2.png'

    zero_shot_plot(data_set=[(x, y)], filename=file, xlabel=xlabel)


def plot_fig4():
    x = ['more lefts',
         'less middles',
         'average',
         'more rights'
         ]

    y = [81.4,
         86.4,
         88.9,
         81.3]

    for i in range(len(y)):
        y[i] /= 100

    xlabel = '# Position of Prompts'
    file = 'zero_shot_fig3.png'

    zero_shot_plot(data_set=[(x, y)], filename=file, xlabel=xlabel)

def plot_tab3_single(x, y_list, legend_list, filename):
    data_set = []
    for y in y_list:
        for i in range(len(y)):
            y[i] /= 100
        data_set.append((x, y))
    xlabel = 'Data Size'
    zero_shot_plot(data_set=data_set, filename=filename, xlabel=xlabel, legend_list=legend_list)

def plot_tab3():
    x = [32, 100, 300, 500, 700]
    legend_list = ["Zecoler", "CodeBERT", "CodeBERTa"]
    # zecoler, CodeBERT, CodeBERTa
    CD_Java_list = [
        [53.3, 63.6, 85.8, 90.8, 95.1],
        [48.2, 55.4, 51.3, 48.8, 48.1],
        [52.8, 51, 53.9, 50.7, 51.6]
    ]
    CD_SC_list = [
        [90.1, 93.9, 94.3, 93.6, 94.4],
        [65.4, 68.7, 69.4, 70.2, 75.3],
        [50, 64.5, 65.0, 68.3, 73.7]
    ]
    CD_Go_list = [
        [52.8, 99.5, 99.3, 99.1, 99.4],
        [57.9, 50.3, 49.5, 71.2, 100],
        [57.9, 53.1, 52, 65.3, 65.9]
    ]

    plot_tab3_single(x, CD_Java_list, legend_list, filename="cd_java_fs.png")
    plot_tab3_single(x, CD_SC_list, legend_list, filename="cd_sc_fs.png")
    plot_tab3_single(x, CD_Go_list, legend_list, filename="cd_go_fs.png")

if __name__ == '__main__':
    # plot_fig3()
    # plot_fig4()
    plot_tab3()
