import matplotlib.pyplot as plt
import numpy as np
import os
import json
import random
from shutil import copyfile


def plot_loss(loss_list_name, title):
    loss_list = np.load(loss_list_name)
    print(len(loss_list))
    plt.figure()
    plt.plot(np.arange(len(loss_list)), loss_list)
    plt.title(loss_list_name + ": " + title)
    plt.show()


def get_url2code(dataset, name='data.jsonl'):
    url_to_code = {}
    with open(os.path.join(dataset, name)) as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            url_to_code[js['idx']] = js['func']
    return url_to_code


def clonedet_dataset_gen(lang, source_name, folder, target_name, target_name2, max_len, size):
    source_dir = os.path.join("../dataset", lang)
    target_dir = os.path.join("../CloneDetection_32dev", folder)
    target_dir2 = os.path.join("../dataset/new", folder)
    url_to_code = get_url2code(source_dir)
    size //= 2
    jsonl_list_p = []
    jsonl_list_n = []
    txt_list_p = []
    txt_list_n = []
    p = 0
    n = 0
    with open(os.path.join(source_dir, source_name)) as f:
        idx = 1
        lines = f.readlines()
    random.shuffle(lines)
    for line in lines:
        if p == size and n == size:
            break
        l = line.split()
        label = True if l[2] == "1" else False
        if label is True and p == size or label is False and n == size:
            continue
        code1 = " ".join(url_to_code[l[0]].split())
        code2 = " ".join(url_to_code[l[1]].split())
        if len(code1) + len(code2) > max_len:
            continue
        if label is True:
            jsonl_list_p.append({
                "code1": code1,
                "code2": code2,
                "idx": idx,
                "label": label,
            })
            txt_list_p.append("\t".join(l))
            p += 1
        else:
            jsonl_list_n.append({
                "code1": code1,
                "code2": code2,
                "idx": idx,
                "label": label,
            })
            txt_list_n.append("\t".join(l))
            n += 1
        idx += 1
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    if not os.path.exists(target_dir2):
        os.makedirs(target_dir2)
    min_c = min(len(txt_list_p), len(txt_list_n))
    jsonl_list_p = jsonl_list_p[:min_c]
    jsonl_list_p.extend(jsonl_list_n[:min_c])
    txt_list_p = txt_list_p[:min_c]
    txt_list_p.extend(txt_list_n[:min_c])
    with open(os.path.join(target_dir, target_name), 'w') as f:
        random.shuffle(jsonl_list_p)
        for item in jsonl_list_p:
            f.write(json.dumps(item) + "\n")
    with open(os.path.join(target_dir2, target_name2), 'w') as f:
        random.shuffle(txt_list_p)
        f.write("\n".join(txt_list_p))
    print(min_c*2)


def reduce(source, target, num):
    js_True_list = []
    js_False_list = []
    with open(source) as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            if js["label"]:
                js_True_list.append(js)
            else:
                js_False_list.append(js)
    num //= 2
    js_list = []
    js_list.extend(random.sample(js_True_list, num))
    js_list.extend(random.sample(js_False_list, num))
    random.shuffle(js_list)
    with open(target, 'w') as f:
        for item in js_list:
            f.write(json.dumps(item) + "\n")


def clean_space(file):
    js_list = []
    with open(file) as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            js["code1"] = " ".join(js["code1"].split())
            js["code2"] = " ".join(js["code2"].split())
            js_list.append(js)
    with open(file, 'w') as f:
        for item in js_list:
            f.write(json.dumps(item) + "\n")


def gen_java_dataset(folder, trainSize):
    clonedet_dataset_gen(lang="Java", source_name="train.txt", folder=folder,
                         target_name="train.jsonl", target_name2="train.txt", max_len=1200, size=trainSize)
    copyfile("../CloneDetection_32dev/Java_static/val.jsonl", "../CloneDetection_32dev/{}/val.jsonl".format(folder))
    copyfile("../CloneDetection_32dev/Java_static/dev32.jsonl", "../CloneDetection_32dev/{}/dev32.jsonl".format(folder))
    copyfile("../dataset/new/Java_static/data.jsonl", "../dataset/new/{}/data.jsonl".format(folder))
    copyfile("../dataset/new/Java_static/valid.txt", "../dataset/new/{}/valid.txt".format(folder))
    copyfile("../dataset/new/Java_static/test.txt", "../dataset/new/{}/test.txt".format(folder))


def gen_0shot_dataset(lang):
    clonedet_dataset_gen(lang=lang, source_name="test.txt", folder=lang,
                         target_name="train.jsonl", target_name2="train.txt", max_len=1200, size=32)
    clonedet_dataset_gen(lang=lang, source_name="test.txt", folder=lang,
                         target_name="dev32.jsonl", target_name2="valid.txt", max_len=1200, size=384)
    clonedet_dataset_gen(lang=lang, source_name="test.txt", folder=lang,
                         target_name="val.jsonl", target_name2="test.txt", max_len=1200, size=960)
    copyfile("../dataset/{}/data.jsonl".format(lang), "../dataset/new/{}/data.jsonl".format(lang))


if __name__ == '__main__':
    # plot_loss("output/clonedet_0801/p10-i0/loss.npy", "loss")
    plot_loss("output/clonedet_0802_f/p10-i0/acc.npy", "acc")
    # gen_0shot_dataset(lang="JavaScript")
    # gen_0shot_dataset(lang="Python")
    # gen_java_dataset(folder="Java_3k", trainSize=3000)
    # clonedet_dataset_gen(lang="Java", source_name="train.txt", folder="Java_static",
    #                      target_name="val.jsonl", target_name2="test.txt", max_len=1200, size=960)
