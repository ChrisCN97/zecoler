import json
import os.path
import random

import matplotlib as mpl
import matplotlib.pyplot as plt


def solidityProcess():
    root = "/mnt/sda/cn/codet5/data/summarize/smartContract/"
    train_name = root + "train.jsonl"
    examples = []
    with open(train_name, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            code = js['code'].split()
            if len(code) < 25 or len(code) > 200:
                continue
            nl = js['summary'].split()
            if len(nl) < 1 or len(nl) > 100:
                continue
            examples.append({"code_tokens": code, "docstring_tokens": nl})
            if idx + 1 == 26000:
                break
    root = "/mnt/sda/cn/codet5/data/summarize/solidity/"
    with open(root + "train.jsonl", 'w') as f:
        for item in examples[:5000]:
            f.write(json.dumps(item) + "\n")
    with open(root + "valid.jsonl", 'w') as f:
        for item in examples[5000:6000]:
            f.write(json.dumps(item) + "\n")
    with open(root + "test.jsonl", 'w') as f:
        for item in examples[6000:]:
            f.write(json.dumps(item) + "\n")

def summarize2src():
    sum_root = "/mnt/sda/cn/codet5/data/summarize"
    pretrain_root = "/mnt/sda/cn/codet5/data/pretrain/src/"
    langs = ["java", "python", "go", "ruby", "javascript", "php", "solidity"]
    for lang in langs:
        sum_name = os.path.join(sum_root, lang, "train.jsonl")
        with open(sum_name, encoding="utf-8") as f, open(pretrain_root + lang + ".txt", 'w') as f2:
            for idx, line in enumerate(f):
                line = line.strip()
                js = json.loads(line)
                code_tokens = js["code_tokens"]
                for i, token in enumerate(code_tokens):
                    if token == '\n':
                        code_tokens[i] = '\\n'
                    if token == '\r':
                        code_tokens[i] = '\\r'
                f2.write(" ".join(code_tokens) + '\n')

def src2with_lang_v1(trg_folder, train_num, val_num):
    pattern = "output language {} . "
    src_folder = "/mnt/sda/cn/codet5/data/pretrain/src/"
    langs = ["java", "python", "go", "ruby", "javascript", "php", "solidity"]
    codes = []
    for lang in langs:
        with open(src_folder + lang + ".txt") as f:
            for line in f:
                codes.append(pattern.format(lang) + line)
    random.shuffle(codes)
    total = len(codes)
    with open(trg_folder + "train.txt", 'w') as f:
        f.write("".join(codes[total-train_num-val_num:total-val_num]))
    with open(trg_folder + "val.txt", 'w') as f:
        f.write("".join(codes[total-val_num:]))

def minimize(file, size):
    examples = []
    with open(file+".backup", encoding="utf-8") as f:
        for line in f:
            examples.append(line)
    random.shuffle(examples)
    with open(file, 'w') as f:
        f.write("".join(examples[:size]))

def min_summarize():
    root = "/mnt/sda/cn/codet5/data/summarize/"
    langs = ["java", "python", "go", "ruby", "javascript", "php", "solidity"]
    for lang in langs:
        file = os.path.join(root, lang, "valid.jsonl")
        minimize(file, 300)
        file = os.path.join(root, lang, "test.jsonl")
        minimize(file, 500)


def case_study(lang):
    sum_root = "/mnt/sda/cn/codet5/data/summarize"
    with open(os.path.join(sum_root, lang, "train.jsonl"), encoding="utf-8") as f:
        lines = f.readlines()
        line = lines[random.randint(0,len(lines)-1)]
        line = line.strip()
        js = json.loads(line)
        func_name = js["func_name"]
        doc = js["docstring"]
        code = js["code"]
        print("method_name: {}\nsummerize: {}\ncode: {}".format(func_name, doc, code))

mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 24
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['axes.linewidth'] = 3  # bounding box line width
file_path = "pics"

def save_fig(fig, filename):
    plt.show()
    file = os.path.join(file_path, filename)
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    fig.savefig(file, dpi=200)

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

def plot_single(x, y_list, legend_list, filename, ylabel):
    data_set = []
    for y in y_list:
        data_set.append((x, y))
    xlabel = 'Data Size'
    zero_shot_plot(data_set=data_set, filename=filename, xlabel=xlabel, legend_list=legend_list, ylabel=ylabel)

def plot_rq2_CM():
    x = [500, 1000, 1500, 2000, 2500]
    # x = [0, 100, 200, 300, 400, 500, 600, 700]
    legend_list = ["Zecoler", "CodeBERT"]

    # SC_list = [
    #     [53.73, 65.33, 68.76, 71.09, 73.44],
    #     [53.26, 64.43, 68.44, 71.31, 72.85]
    # ]
    # Go_list = [
    #     [10.09, 10.51, 11.04, 11.00, 11.06],
    #     [9.91, 10.41, 10.71, 10.89, 10.99]
    # ]

    SC_list = [
        [13.37,18.47,23.20,41.76,48.03,53.73,56.91,59.84],
        [12.68,16.32,21.79,41.24,47.02,53.26,55.88,59.41]
    ]
    Go_list = [
        [8.67,9.12,9.10,9.70,10.29,10.09,10.10,10.22],
        [8.21,8.53,8.85,9.21,9.49,9.91,9.67,9.83]
    ]

    plot_single(x, SC_list, legend_list, filename="cm_sc_rq2.png", ylabel="BLEU")
    plot_single(x, Go_list, legend_list, filename="cm_go_rq2.png", ylabel="BLEU")

def plot_rq2_CG():
    x = [500, 1000, 1500, 2000, 2500]
    # x = [0,100, 200, 300, 400, 500, 600, 700]
    legend_list = ["Zecoler", "CodeBERT"]

    # SC_list = [
    #     [27.30, 43.53, 49.58, 52.39, 55.20],
    #     [26.14, 42.55, 48.12, 51.06, 55.43]
    # ]
    # Go_list = [
    #     [8.67, 13.07, 14.82, 14.94, 15.50],
    #     [6.24, 12.02, 14.68, 15.50, 16.18]
    # ]
    SC_list = [
        [3.20,1.69,5.20,12.99,21.87,27.30,32.98,34.03],
        [2.74,1.06,3.39,9.50,17.07,26.14,32.81,34.63]
    ]
    Go_list = [
        [9.18,5.09,5.28,5.68,6.53,8.67,9.16,10.28],
        [8.68,6.64,5.89,6.37,6.19,6.24,7.68,10.05]
    ]

    plot_single(x, SC_list, legend_list, filename="cg_sc_rq2.png", ylabel="BLEU")
    plot_single(x, Go_list, legend_list, filename="cg_go_rq2.png", ylabel="CodeBLEU")

def plot_rq3_CM():
    x = [500, 1000, 1500, 2000, 2500]
    legend_list = ["Zecoler", "CodeBERT"]

    # SC_list = [
    #     [26.82, 46.22, 60.13, 64.26, 68.30],
    #     [26.69,46.10,59.49,65.00,67.26]
    # ]
    # Go_list = [
    #     [8.90, 9.40, 9.54, 9.60, 10.10],
    #     [8.69,8.69,9.34,8.92,10.31]
    # ]

    SC_list = [
        [38.64, 58.89, 67.44, 69.84, 71.29],
        [38.79, 58.17, 66.87, 69.70, 71.08]
    ]
    Go_list = [
        [8.90, 9.40, 9.54, 9.60, 10.10],
        [8.69, 8.69, 9.34, 8.92, 10.31]
    ]

    plot_single(x, SC_list, legend_list, filename="cm_sc_rq3.png", ylabel="BLEU")
    plot_single(x, Go_list, legend_list, filename="cm_go_rq3.png", ylabel="BLEU")

def plot_rq3_CG():
    x = [500, 1000, 1500, 2000, 2500]
    legend_list = ["Zecoler", "CodeBERT"]

    # SC_list = [
    #     [17.68, 25.21, 34.65, 44.69, 47.53],
    #     [15.52,23.69,31.57,44.17,46.37]
    # ]
    # Go_list = [
    #     [10.55,8.22,8.46,10.06,12.69],
    #     [4.97,7.85,9.68,10.36,11.01]
    # ]

    SC_list = [
        [21.36,32.19,45.64,48.49,50.90],
        [19.72,33.44,45.95,51.04,52.75]
    ]
    Go_list = [
        [10.55, 8.22, 8.46, 10.06, 12.69],
        [4.97, 7.85, 9.68, 10.36, 11.01]
    ]

    plot_single(x, SC_list, legend_list, filename="cg_sc_rq3.png", ylabel="BLEU")
    plot_single(x, Go_list, legend_list, filename="cg_go_rq3.png", ylabel="CodeBLEU")


if __name__ == "__main__":
    # trg_folder = "/mnt/sda/cn/codet5/data/pretrain/with_lang/v1/"
    # trg_folder = "/mnt/sda/cn/codet5/data/pretrain/test/"
    # train_num = 5000
    # val_num = 500
    # src2with_lang_v1(trg_folder, train_num, val_num)
    # min_summarize()
    # plot_rq2_CM()
    # plot_rq2_CG()
    # plot_rq3_CM()
    # plot_rq3_CG()
    case_study(lang="go")
