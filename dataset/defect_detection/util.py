import json
import os
import random
import csv
import shutil

import numpy as np
from matplotlib import pyplot as plt

CODENET_PATH = "/mnt/sda/cn/python/codeBert/codeNet/Project_CodeNet"

def get_random_problem():
    p_list = os.listdir(os.path.join(CODENET_PATH, "metadata"))
    return random.choice(p_list)

def code_preprocess(code_content):
    return " ".join(code_content.split())

def get_submission_list(p_name, lang, status, code_lem_min=300, code_lem_max=1200, limit=-1):
    with open(os.path.join(CODENET_PATH, "metadata", p_name), newline='') as f:
        reader = csv.DictReader(f)
        sub_list = []
        for row in reader:
            if "language" not in row:
                break
            if row["language"] == lang:
                for s in status:
                    if row["status"] == s:
                        s_name = "{}.{}".format(row["submission_id"], row["filename_ext"])
                        with open(os.path.join(CODENET_PATH, "data", p_name.split(".")[0], lang, s_name)) as f:
                            code = code_preprocess(f.read())
                            if code_lem_min < len(code) < code_lem_max:
                                sub_list.append((s_name, code))
                                if limit != -1 and len(sub_list) >= limit:
                                    return sub_list
    return sub_list

def get_random_submission(promblem, lang, status):
    status = [status]
    s_list = get_submission_list(promblem, lang, status)
    if s_list:
        return random.choice(s_list)
    else:
        return None

def example(lang, status):
    submission = None
    while not submission:
        p_name = get_random_problem()
        submission = get_random_submission(p_name, lang, status)
    print(submission[1])

def get_clear_folder(lang):
    if os.path.exists(lang):
        shutil.rmtree(lang)
    os.mkdir(lang)
    return lang

def gen_data_jsonl(data_jsonl_list, folder):
    jsonl_path = os.path.join(folder, "data.jsonl")
    with open(jsonl_path, 'w') as f:
        for item in data_jsonl_list:
            f.write(json.dumps(item) + "\n")

def get_qualified_submission_id_list(lang, status_list, code_lem_min, code_lem_max, limit=-1, max_size=50000):
    p_list = os.listdir(os.path.join(CODENET_PATH, "metadata"))
    dataset_list = []
    data_jsonl_list = []
    data_jsonl_list.append({"func": ".", "idx": "empty"})
    total_p_num = len(p_list)
    idx = 1
    for p_name in p_list[:limit]:
        if len(dataset_list) >= max_size:
            break
        if idx % 50 == 1:
            print("Check {}/{}, current size: {}".format(idx, total_p_num, len(dataset_list)))
        idx += 1
        n_s_list = get_submission_list(p_name, lang, status_list, code_lem_min, code_lem_max)
        if n_s_list:
            p_s_list = get_submission_list(p_name, lang, ["Accepted"],
                                           code_lem_min, code_lem_max, limit=len(n_s_list))
            if len(n_s_list) > len(p_s_list):
                n_s_list = random.sample(n_s_list, len(p_s_list))
            for name, code in p_s_list:
                dataset_list.append("\t".join([name, "empty", "1"]))
                data_jsonl_list.append({"func": code, "idx": name})
            for name, code in n_s_list:
                dataset_list.append("\t".join([name, "empty", "0"]))
                data_jsonl_list.append({"func": code, "idx": name})
    folder = get_clear_folder(lang)
    gen_data_jsonl(data_jsonl_list, folder)
    size_txt = "total_{}.txt".format(len(dataset_list))
    with open(os.path.join(folder, size_txt), 'w') as f:
        random.shuffle(dataset_list)
        f.write("\n".join(dataset_list))
    print("data.jsonl: {}, {}".format(len(data_jsonl_list), size_txt))
    return size_txt

def check_trainrepeat_pnrate(lang, name):
    url = {}
    with open(os.path.join(lang, 'data.jsonl')) as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            url[js['idx']] = 0
    txt_path = os.path.join(lang, name)
    with open(txt_path) as f:
        line = f.readline()
        while line is not None and line != '':
            l = line.split()
            url[l[0]] += 1
            line = f.readline()
    url = sorted(url.items(), key=lambda item: item[1], reverse=True)
    count_list = []
    for c in url:
        count_list.append(c[1])
    count_list = np.array(count_list)
    count_list = count_list[count_list != 0]
    plt.plot(np.arange(len(count_list)), count_list)
    plt.show()
    print("Repeat: Max:{}, Min:{}, Average:{}".format(count_list[0], count_list[-1], count_list.mean()))
    positive = 0
    total = 0
    with open(txt_path, 'r') as f:
        for line in f.readlines():
            line = line.split()
            if line[2] == "1":
                positive += 1
            total += 1
    print("Pnrate: total:{} positive:{} rate:{:.3f}".format(total, positive, positive / total))

def get_url2code(lang, name='data.jsonl'):
    url_to_code = {}
    with open(os.path.join(lang, name)) as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            url_to_code[js['idx']] = js['func']
    return url_to_code

def check_example(lang, name):
    url_to_code = get_url2code(lang)
    with open(os.path.join(lang, name)) as f:
        sample_list = random.choice(f.readlines()[:100]).split()
    print("*"*20)
    print(url_to_code[sample_list[0]])
    print("*" * 20)
    if sample_list[2] == "1":
        print("Defect!")
    else:
        print("Not Defect!")

def split_txt(lang, source, target1, size, target2=""):
    with open(os.path.join(lang, source)) as sf:
        with open(os.path.join(lang, "{}_{}.txt".format(target1, size)), 'w') as t1f:
            sl = [s.strip() for s in sf.readlines()]
            if size > len(sl):
                print("{}.txt does not contain {} data".format(source, size))
                return
            random.shuffle(sl)
            p_num = size // 2
            n_num = p_num
            idx = len(sl)-1
            t1_ls = []
            while True:
                if sl[idx].split()[2] == "1":
                    if p_num > 0:
                        t1_ls.append(sl[idx])
                        sl.pop(idx)
                        p_num -= 1
                else:
                    if n_num > 0:
                        t1_ls.append(sl[idx])
                        sl.pop(idx)
                        n_num -= 1
                idx -= 1
                if p_num == 0 and n_num == 0:
                    break
            random.shuffle(t1_ls)
            t1f.write("\n".join(t1_ls))
    if target2 != "":
        with open(os.path.join(lang, "{}_{}.txt".format(target2, len(sl))), 'w') as t2f:
            random.shuffle(sl)
            t2f.write("\n".join(sl))
        return "{}_{}.txt".format(target2, len(sl))
    else:
        return ""

def gen_new_lang(lang, status_list, code_len_min, code_len_max):
    size_txt = get_qualified_submission_id_list(lang=lang, status_list=status_list,
                                                code_lem_min=code_len_min, code_lem_max=code_len_max)
    train_txt = split_txt(lang, source=size_txt, target1="temp", size=1400, target2="train")
    split_txt(lang, source="temp_1400.txt", target1="dev", size=400, target2="test")
    os.system("rm {}/temp_1400.txt".format(lang))
    os.system("mv {0}/{1} {0}/train.txt".format(lang, train_txt))

def gen_train(lang, size):
    split_txt(lang=lang, source="train.txt", target1="train", size=size)

def check_repeat(lang, s1, s2):
    with open(os.path.join(lang, s1)) as f1:
        f1_set = set()
        for line in f1:
            line = line.split()
            f1_set.add(line[0])
    c = 0
    with open(os.path.join(lang, s2)) as f2:
        for line in f2:
            s = line.split()[0]
            if s in f1_set:
                c += 1
    print(c)

"""
Compile Error          | CE  |  0
Memory Limit Exceeded  | MLE |  3
Runtime Error          | RE  |  7
Accepted               | AC  |  4
"""
langs = ["Java", "Python", "JavaScript", "PHP", "Ruby", "Go", "C#", "C++", "C", "Haskell", "Kotlin", "Fortran"]

if __name__ == "__main__":
    status_list = ["Compile Error", "Runtime Error"]
    # lang = "Java"
    # example(lang="Java", status="Runtime Error")
    # size_txt = get_qualified_submission_id_list(lang="Java", status_list=status_list,
    #                                  code_lem_min=300, code_lem_max=600, limit=-1)
    # check_trainrepeat_pnrate(lang="Java", name=size_txt)
    # check_example(lang="Java", name="total_76662.txt")
    # check_repeat(lang="Java", s1="train.txt", s2="train_32.txt")
    # for lang in ["Python", "JavaScript", "PHP", "Ruby", "Go", "C#", "C++", "C", "Haskell", "Kotlin", "Fortran"]:
    #         gen_train(lang=lang, size=32)
    # gen_train(lang="Java", size=10000)
