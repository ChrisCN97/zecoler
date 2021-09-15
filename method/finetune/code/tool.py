import matplotlib.pyplot as plt
import numpy as np
import json
import os
import random
import re

def plot_loss(loss_list_name):
    loss_list = np.load(loss_list_name)
    print(len(loss_list))
    plt.figure()
    plt.plot(np.arange(len(loss_list)), loss_list)
    plt.title(loss_list_name)
    plt.show()

def check_data(name):
    with open(name, 'r') as json_file:
        json_list = list(json_file)
        result = [json.loads(jline) for jline in json_list]
        func_len = 0
        for i in result:
            func_len += len(i["func"].split())
        func_len /= len(result)
        print(func_len)

def get_url2code(dataset, name='data.jsonl'):
    url_to_code = {}
    with open(os.path.join(dataset, name)) as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            url_to_code[js['idx']] = js['func']
    return url_to_code

def check_example(dataset, name):
    url_to_code = get_url2code(dataset)
    with open(os.path.join(dataset, "{}.txt".format(name))) as f:
        sample_list = random.choice(f.readlines()[:100]).split()
    print("*"*20)
    print(url_to_code[sample_list[0]])
    print("*" * 20)
    print(url_to_code[sample_list[1]])
    print("*" * 20)
    if sample_list[2] == "1":
        print("Clone!")
    else:
        print("Not Clone!")

def file_pnrate(name):
    positive = 0
    total = 0
    with open(name, 'r') as f:
        for line in f.readlines():
            line = line.split()
            if line[2] == "1":
                positive += 1
            total += 1
    print("total:{} positive:{} rate:{:.3f}".format(total, positive, positive/total))
"""
def get_s(data_path, p, lang, id_set, data_jsonl_list):
    s_list_folder = os.path.join(data_path, p, lang)
    if os.path.exists(s_list_folder):
        s_list = os.listdir(s_list_folder)
        if len(s_list) == 0:
            return "-1"
        s = random.choice(s_list)
        if s not in id_set:
            s_path = os.path.join(s_list_folder, s)
            with open(s_path) as f:
                s_content = f.read()
            s_len = len(s_content.split())
            if s_len<60 or s_len>240:
                return "-1"
            id_set.add(s)
            data_jsonl_list.append({"func": s_content, "idx": s})
        return s
    return "-1"

def extract_dataset(lang, size, pos_rate, data_path, dataset_path, name, append):
    print("Extract {}.txt...".format(name))
    p_list = os.listdir(data_path)
    size_c = 0
    neg_num = int((1-pos_rate)/pos_rate)
    data_jsonl_list = []
    id_set = set()
    jsonl_path = os.path.join(dataset_path, "data.jsonl")
    if append and os.path.exists(jsonl_path):
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                js = json.loads(line)
                data_jsonl_list.append(js)
                id_set.add(js["idx"])
    dataset_list = []
    round_num = 1
    while size_c < size:
        p = random.choice(p_list)
        s = get_s(data_path, p, lang, id_set, data_jsonl_list)
        if s == "-1":
            continue
        s_p = get_s(data_path, p, lang, id_set, data_jsonl_list)
        if s_p == s or s_p == "-1":
            continue
        dataset_list.append("\t".join([s, s_p, "1"]))
        neg = 0
        while neg < neg_num:
            while True:
                p_n = random.choice(p_list)
                if p != p_n:
                    break
            s_n = get_s(data_path, p_n, lang, id_set, data_jsonl_list)
            if s_n == "-1":
                continue
            dataset_list.append("\t".join([s, s_n, "0"]))
            neg += 1
        size_c += 1 + neg_num
        if size_c >= round_num*1000:
            print("{}/{}".format(size_c, size))
            round_num += 1
    with open(jsonl_path, 'w') as f:
        for item in data_jsonl_list:
            f.write(json.dumps(item) + "\n")
    with open(os.path.join(dataset_path, "{}.txt".format(name)), 'w') as f:
        random.shuffle(dataset_list)
        f.write("\n".join(dataset_list))

def get_full_dataset(lang, size_b, size_s, dataset_path, train):
    print(dataset_path)
    if train:
        extract_dataset(lang=lang, size=size_b, pos_rate=0.5,
                        data_path="/mnt/sda/cn/python/codeBert/codeNet/Project_CodeNet/data",
                        dataset_path=dataset_path, name="train", append=False)
    extract_dataset(lang=lang, size=size_s, pos_rate=0.5,
                    data_path="/mnt/sda/cn/python/codeBert/codeNet/Project_CodeNet/data",
                    dataset_path=dataset_path, name="valid", append=True if train else False)
    extract_dataset(lang=lang, size=size_s, pos_rate=0.5,
                    data_path="/mnt/sda/cn/python/codeBert/codeNet/Project_CodeNet/data",
                    dataset_path=dataset_path, name="test", append=True)
"""

java_need_num = 950
other_need_num = 200
langs_need = ["Java", "Python", "Ruby", "C#", "C++"]
code_len_min = 55
code_len_max = 190

def check_problem(limit=-1):
    p_list = os.listdir("/mnt/sda/cn/python/codeBert/codeNet/Project_CodeNet/data")
    p_qualified_list = []
    p_total = len(p_list)
    p_cur = 0
    for p_id in p_list[:limit]:
        # print(p_id+": ", end="")
        p_cur += 1
        p_folder_path = os.path.join("/mnt/sda/cn/python/codeBert/codeNet/Project_CodeNet/data", p_id)
        p_langs = os.listdir(p_folder_path)
        lack_lang = False
        for lang in langs_need:
            if lang not in p_langs:
                # print("Lack of "+lang)
                lack_lang = True
                break
        if lack_lang:
            continue
        p_qualified = True
        for lang in langs_need:
            lang_path = os.path.join(p_folder_path, lang)
            codes = os.listdir(lang_path)
            if lang == "Java":
                if len(codes) < java_need_num:
                    # print("Lack code in {}, codes num: {}".format(lang, len(codes)))
                    p_qualified = False
                    break
            else:
                if len(codes) < other_need_num:
                    # print("Lack code in {}, codes num: {}".format(lang, len(codes)))
                    p_qualified = False
                    break
            codes_qualified = 0
            for code_name in codes:
                code_path = os.path.join(lang_path, code_name)
                with open(code_path) as f:
                    code_content = f.read()
                code_len = len(code_content.split())
                if code_len_min < code_len < code_len_max:
                    codes_qualified += 1
                if lang == "Java":
                    if codes_qualified == java_need_num:
                        break
                else:
                    if codes_qualified == other_need_num:
                        break
            if lang == "Java":
                if codes_qualified < java_need_num:
                    # print("Lack code in {}, qualified codes num: {}, total: {}".format(lang, codes_qualified, len(codes)))
                    p_qualified = False
                    break
            else:
                if codes_qualified < other_need_num:
                    # print("Lack code in {}, qualified codes num: {}, total: {}".format(lang, codes_qualified, len(codes)))
                    p_qualified = False
                    break
        if p_qualified:
            print("check: {}/{} {} Qualified.".format(p_cur, p_total, p_id))
            p_qualified_list.append(p_id)
            if len(p_qualified_list) == 10:
                break
    # print("check: {}/{}".format(p_cur, p_total))
    print("Num: "+str(len(p_qualified_list)))
    print(p_qualified_list)

p_list = ['p02900', 'p02411', 'p02641', 'p02756', 'p02761', 'p03471', 'p02959', 'p03087', 'p02409', 'p02623']
dataset_folder = "../dataset"
def extract_code(code_need_num, lang):
    p_code_list = []
    for p_id in p_list:
        code_folder_path = os.path.join("/mnt/sda/cn/python/codeBert/codeNet/Project_CodeNet/data", p_id, lang)
        if not os.path.exists(code_folder_path):
            continue
        codes = os.listdir(code_folder_path)
        code_list = []
        code_num = 0
        for code_name in codes:
            code_path = os.path.join(code_folder_path, code_name)
            with open(code_path) as f:
                code_content = f.read()
            code_len = len(code_content.split())
            if code_len_min < code_len < code_len_max:
                code_list.append((code_name, code_content))
                code_num += 1
                if code_num >= code_need_num:
                    break
        p_code_list.append(code_list)
    return p_code_list

def get_folder(lang):
    folder = os.path.join(dataset_folder, lang)
    if not os.path.exists(folder):
        os.mkdir(folder)
    return folder

def gen_data_jsonl(p_code_list, lang):
    folder = get_folder(lang)
    data_jsonl_list = []
    jsonl_path = os.path.join(folder, "data.jsonl")
    for p in p_code_list:
        for code in p:
            data_jsonl_list.append({"func": code[1], "idx": code[0]})
    with open(jsonl_path, 'w') as f:
        for item in data_jsonl_list:
            f.write(json.dumps(item) + "\n")

def gen_pair(lang, name, p_code_list, pcn, ncn):
    """
    :param pcn: positive code num
    :param ncn: negative code num
    """
    print("Generate pair for {}/{}".format(lang, name))
    dataset_list = []
    repeat_check_set = set()
    folder = get_folder(lang)
    p_num = len(p_code_list)
    for i in range(p_num):
        c_num = len(p_code_list[0])
        for idx, code in enumerate(p_code_list[i]):
            # print("Problem: {}/{}, Code: {}/{}".format(i+1, p_num, idx+1, c_num))
            for n in range(pcn):
                repeat_num = 0
                while True:
                    if repeat_num >= 3:
                        break
                    repeat_num += 1
                    p_code = random.choice(p_code_list[i])
                    if p_code is not code:
                        pair = "\t".join([code[0], p_code[0], "1"])
                        if pair not in repeat_check_set:
                            dataset_list.append(pair)
                            repeat_check_set.add(pair)
                            break
            for n in range(ncn):
                repeat_num = 0
                while True:
                    if repeat_num >= 3:
                        break
                    repeat_num += 1
                    n_p = random.choice(p_code_list)
                    if n_p is not p_code_list[i]:
                        if len(n_p) == 0:
                            continue
                        n_code = random.choice(n_p)
                        pair = "\t".join([code[0], n_code[0], "0"])
                        if pair not in repeat_check_set:
                            dataset_list.append(pair)
                            repeat_check_set.add(pair)
                            break
    with open(os.path.join(folder, "{}.txt".format(name)), 'w') as f:
        random.shuffle(dataset_list)
        f.write("\n".join(dataset_list))
    return len(dataset_list)

def extract_java_train_valid_test(code_need_num=950, train_num=750, valid_num=100):
    p_code_list = extract_code(code_need_num, "Java")
    gen_data_jsonl(p_code_list, "Java")
    train_code_list = [code_list[:train_num] for code_list in p_code_list]
    valid_code_list = [code_list[train_num:train_num + valid_num] for code_list in p_code_list]
    test_code_list = [code_list[train_num + valid_num:] for code_list in p_code_list]
    gen_pair(lang="Java", name="train", p_code_list=train_code_list, pcn=60, ncn=60)
    gen_pair(lang="Java", name="valid", p_code_list=valid_code_list, pcn=20, ncn=20)
    gen_pair(lang="Java", name="test", p_code_list=test_code_list, pcn=20, ncn=20)

def extract_valid_test(lang, code_need_num=200, valid_num=100, pcn=20, ncn=20):
    p_code_list = extract_code(code_need_num, lang)
    gen_data_jsonl(p_code_list, lang)
    test_code_list = [code_list[:valid_num] for code_list in p_code_list]
    valid_code_list = [code_list[valid_num:] for code_list in p_code_list]
    test_num = gen_pair(lang=lang, name="test", p_code_list=test_code_list, pcn=pcn, ncn=ncn)
    valid_num = gen_pair(lang=lang, name="valid", p_code_list=valid_code_list, pcn=pcn, ncn=ncn)
    print("Valid: {}, Test: {}".format(valid_num, test_num))

def extract_dataset():
    extract_java_train_valid_test()
    for lang in langs_need[1:]:
        extract_valid_test(lang)

def check_frequency(folder):
    url = {}
    with open(os.path.join('../dataset', folder, 'data.jsonl')) as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            url[js['idx']] = 0
    with open(os.path.join('../dataset', folder, "train.txt")) as f:
        line = f.readline()
        while line is not None and line != '':
            l = line.split()
            url[l[0]] += 1
            url[l[1]] += 1
            line = f.readline()
    url = sorted(url.items(), key=lambda item: item[1], reverse=True)
    count_list = []
    for c in url:
        count_list.append(c[1])
    count_list = np.array(count_list)
    count_list = count_list[count_list != 0]
    plt.plot(np.arange(len(count_list)), count_list)
    plt.show()
    print("Total_Idx:{}, Max:{}, Min:{}, Average:{}".format(len(count_list), count_list[0], count_list[-1], count_list.mean()))

def check_repeat(folder):
    train_set = set()
    with open(os.path.join('../dataset', folder, "train.txt")) as f:
        line = f.readline()
        while line is not None and line != '':
            l = line.split()
            train_set.update(l[0], l[1])
            line = f.readline()

    url = {}
    with open(os.path.join('../dataset', folder, "test.txt")) as f:
        line = f.readline()
        while line is not None and line != '':
            l = line.split()
            if l[0] not in train_set:
                url[l[0]] = 0
            else:
                if l[0] not in url:
                    url[l[0]] = 1
                else:
                    url[l[0]] += 1
            if l[1] not in train_set:
                url[l[1]] = 0
            else:
                if l[1] not in url:
                    url[l[1]] = 1
                else:
                    url[l[1]] += 1
            line = f.readline()

    url = sorted(url.items(), key=lambda item: item[1], reverse=True)
    count_list = []
    repeat = 0
    for c in url:
        if c[1] > 0:
            repeat += 1
        count_list.append(c[1])
    count_list = np.array(count_list)
    # plt.plot(np.arange(len(count_list)), count_list)
    # plt.show()
    print("Total_Idx:{}, Max_Repeat:{}, Repeat_Rate:{}".format(len(url), count_list[0], repeat/len(url)))

def reduce(s, t):
    with open("../dataset/{}/train.txt".format(s)) as f:
        with open("../dataset/{}/train.txt".format(t), 'w') as fw:
            fl = f.readlines()
            fw.write("".join(fl))
    with open("../dataset/{}/valid.txt".format(s)) as f:
        with open("../dataset/{}/valid.txt".format(t), 'w') as fw:
            fl = f.readlines()
            fw.write("".join(random.sample(fl, 384)))
    with open("../dataset/{}/test.txt".format(s)) as f:
        with open("../dataset/{}/test.txt".format(t), 'w') as fw:
            fl = f.readlines()
            fw.write("".join(random.sample(fl, 960)))

def reduce_test(folder, num):
    with open("../dataset/{}/test.txt".format(folder)) as f:
        fl = f.readlines()
    with open("../dataset/{}/test_s.txt".format(folder), 'w') as f:
        f.write("".join(fl))
    with open("../dataset/{}/test.txt".format(folder), 'w') as f:
        f.write("".join(random.sample(fl, num)))

def java_preprocess():
    url_to_code = get_url2code("../dataset/Java_r", name="data_old.jsonl")
    data_jsonl_list = []
    p_len = len("public static void main(String[] args)")
    for idx, func in url_to_code.items():
        i = func.find("public static void main(String[] args)")
        if i != -1:
            func = func[i+p_len:]
            func = re.sub(r"//.*\n", " ", func)
        data_jsonl_list.append({"func": func, "idx": idx})
    with open("../dataset/Java_r/data.jsonl", 'w') as f:
        for item in data_jsonl_list:
            f.write(json.dumps(item) + "\n")

def check_false_example(lang, save, limit):
    dataset_path = os.path.join("../dataset", lang)
    url_to_code = get_url2code(dataset_path)
    with open(os.path.join(dataset_path, "test.txt")) as f:
        test_list = [l.split() for l in f.readlines()]
    with open(os.path.join(save, "pre_{}.txt".format(lang))) as f:
        pre_list = [l.split() for l in f.readlines()]
    count = 0
    for test, pre in zip(test_list, pre_list):
        if test[2] != pre[2]:
            print("*" * 20)
            print(url_to_code[test[0]])
            print("*" * 20)
            print(url_to_code[test[1]])
            print("*" * 20)
            if test[2] == "1":
                print("Test: Clone!, Pre: Not Clone!")
            else:
                print("Test: Not Clone!, Pre: Clone!")
            count += 1
            if count >= limit:
                break


if __name__ == '__main__':
    # plot_loss("./save0802/loss.npy")
    plot_loss("./save0802_f/acc.npy")
    # reduce(s="Java", t="Java_eval")
