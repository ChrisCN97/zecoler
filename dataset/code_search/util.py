import csv
import json
import os
import random
import re
import shutil
import numpy as np
from matplotlib import pyplot as plt

CODENET_PATH = "/mnt/sda/cn/python/codeBert/codeNet/Project_CodeNet"

def txt_preprocess(content, idx=-1, max_len=1100, min_len=10):
    txt_list = re.findall(r">(.+?)<", content, re.DOTALL)
    txt = ""
    for t in txt_list:
        t = " ".join(t.split())
        if t == "Input" or "Sample Input" in t or "Input Format" == t or t == "Input:":
            break
        if "入" in t or t == "Problem is available from here." or t == "Problem 9999":
            txt = ""
            break
        txt += t + " "
    txt = " ".join(txt.split())
    if len(txt) > max_len or len(txt) < min_len:
        txt = ""
    return txt

def code_preprocess(code_content):
    return " ".join(code_content.split())

def get_problem_des_list():
    return os.listdir(os.path.join(CODENET_PATH, "problem_descriptions"))

def get_qualified_prob_list():
    total_len = 0
    pd_list = get_problem_des_list()
    list_len = len(pd_list)
    q_p_list = []
    for idx, pd in enumerate(pd_list):
        with open(os.path.join(CODENET_PATH, "problem_descriptions", pd)) as f:
            txt = txt_preprocess(f.read(), idx)
            if txt == "":
                list_len -= 1
            else:
                total_len += len(txt)
                q_p_list.append(pd)
    print("Problem: {}, average len: {}".format(list_len, total_len / list_len))
    np.save("problems.npy", np.array(q_p_list))

def get_qualified_code(problem, lang, max_size, code_lem_min=300, code_lem_max=600):
    with open(os.path.join(CODENET_PATH, "metadata", problem+".csv"), newline='') as f:
        reader = csv.DictReader(f)
        sub_list = []
        for row in reader:
            if "language" not in row:
                break
            if row["language"] == lang:
                if row["status"] == "Accepted":
                    s_name = "{}.{}".format(row["submission_id"], row["filename_ext"])
                    with open(os.path.join(CODENET_PATH, "data", problem, lang, s_name)) as f:
                        code = code_preprocess(f.read())
                        if code_lem_min < len(code) < code_lem_max:
                            sub_list.append((s_name, code))
                            if len(sub_list) >= max_size*2:
                                break
    if len(sub_list) > max_size:
        sub_list = random.sample(sub_list, max_size)
    return sub_list

def get_sub_random(pp, p_list, lang, code_lem_min=300, code_lem_max=600):
    while True:
        p = pp
        while p == pp:
            p = random.choice(p_list)
        folder = os.path.join(CODENET_PATH, "data", p.split(".")[0], lang)
        if not os.path.exists(folder):
            continue
        s_list = os.listdir(folder)
        if not s_list:
            continue
        c = 0
        while True:
            if c > 3:
                break
            s = random.choice(s_list)
            with open(os.path.join(folder, s)) as f:
                code = code_preprocess(f.read())
                if code_lem_min < len(code) < code_lem_max:
                    return s, code
            c += 1

def get_clear_folder(lang):
    if os.path.exists(lang):
        shutil.rmtree(lang)
    os.makedirs(lang)
    return lang

def gen_data_jsonl(data_jsonl_list, folder):
    jsonl_path = os.path.join(folder, "data.jsonl")
    with open(jsonl_path, 'w') as f:
        for item in data_jsonl_list:
            f.write(json.dumps(item) + "\n")

def gen_new_lang_dataset(lang, max_size, code_lem_min=300, code_lem_max=600):
    p_list = np.load("problems.npy").tolist()
    dataset_list = []
    data_jsonl_list = []
    code_set = set()
    p_list_len = len(p_list)
    for idx, p in enumerate(p_list):
        print("{} Problem: {}/{}".format(lang, idx, p_list_len))
        idx = str(idx)
        with open(os.path.join(CODENET_PATH, "problem_descriptions", p)) as f:
            nl = txt_preprocess(f.read())
            data_jsonl_list.append({"func": nl, "idx": idx})
        problem = p.split(".")[0]
        sub_list = get_qualified_code(problem, lang, max_size)
        for name, code in sub_list:
            dataset_list.append("\t".join([idx, name, "1"]))
            data_jsonl_list.append({"func": code, "idx": name})
            code_set.add(name)
            name, code = get_sub_random(p, p_list, lang)
            dataset_list.append("\t".join([idx, name, "0"]))
            if name not in code_set:
                data_jsonl_list.append({"func": code, "idx": name})
                code_set.add(name)
    folder = get_clear_folder(lang)
    gen_data_jsonl(data_jsonl_list, folder)
    size_txt = "total_{}.txt".format(len(dataset_list))
    with open(os.path.join(folder, size_txt), 'w') as f:
        random.shuffle(dataset_list)
        f.write("\n".join(dataset_list))
    print("data.jsonl: {}, {}".format(len(data_jsonl_list), size_txt))
    return size_txt

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
    print(url_to_code[sample_list[1]])
    print("*" * 20)
    if sample_list[2] == "1":
        print("Match!")
    else:
        print("Not Match!")

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

def gen_new_lang(lang, max_size, code_len_min=300, code_len_max=600):
    size_txt = gen_new_lang_dataset(lang, max_size)
    train_txt = split_txt(lang, source=size_txt, target1="temp", size=1400, target2="train")
    split_txt(lang, source="temp_1400.txt", target1="dev", size=400, target2="test")
    os.system("rm {}/temp_1400.txt".format(lang))
    os.system("mv {0}/{1} {0}/train.txt".format(lang, train_txt))

def gen_train(lang, size):
    split_txt(lang=lang, source="train.txt", target1="train", size=size)

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

def python_code_preprocess(code):
    code = re.sub(r"#.*", '', code)
    code = re.sub(r"'''.*'''", '', code, flags=re.S)
    code = re.sub(r"\"\"\".*\"\"\"", '', code, flags=re.S)
    code = code_preprocess(code)
    return code

def process_CSN():
    dataset_list = []
    data_jsonl_list = []
    pair_list = []
    idx = 1
    with open("CSN/train.jsonl") as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            code = python_code_preprocess(js['code'])
            doc = code_preprocess(js['docstring'])
            if 300 < len(code) < 600 and 300 < len(doc) < 600:
                pair_list.append((idx, idx+1))
                dataset_list.append("\t".join([str(idx), str(idx+1), "1"]))
                data_jsonl_list.append({"func": doc, "idx": str(idx)})
                data_jsonl_list.append({"func": code, "idx": str(idx+1)})
                idx += 2
    for doc_i, code_i in pair_list:
        while True:
            d2, c2 = random.choice(pair_list)
            if c2 != code_i:
                break
        dataset_list.append("\t".join([str(doc_i), str(c2), "0"]))
    folder = "CSN"
    gen_data_jsonl(data_jsonl_list, folder)
    size_txt = "total_{}.txt".format(len(dataset_list))
    with open(os.path.join(folder, size_txt), 'w') as f:
        random.shuffle(dataset_list)
        f.write("\n".join(dataset_list))
    print("data.jsonl: {}, {}".format(len(data_jsonl_list), size_txt))
    lang = "CSN"
    train_txt = split_txt(lang, source=size_txt, target1="temp", size=1400, target2="train")
    split_txt(lang, source="temp_1400.txt", target1="dev", size=400, target2="test")
    os.system("rm {}/temp_1400.txt".format(lang))
    os.system("mv {0}/{1} {0}/train.txt".format(lang, train_txt))


if __name__ == "__main__":
    # get_qualified_prob_list()
    # lang = "Java"
    # for size in [32,100,300,500,700,1000,3000,5000,7000,10000]:
    #     gen_train(lang="SC", size=size)
    # langs = ["Java", "Python", "JavaScript", "PHP", "Ruby", "Go", "C#", "C++", "C", "Haskell", "Kotlin", "Fortran"]
    # for lang in langs:
    #     gen_train(lang, size=32)
    # check_trainrepeat_pnrate(lang="SC", name="train.txt")
    check_example(lang="SC", name="train.txt")
