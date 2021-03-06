import os
import json
import random
import shutil
import matplotlib.pyplot as plt
import numpy as np

CODENET_PATH = "/mnt/sda/cn/python/codeBert/codeNet/Project_CodeNet/data"

def code_preprocess(code_content):
    return " ".join(code_content.split())

def find_qualified_problem(MIN_SIZE, langs_need, q_need, code_len_min, code_len_max, limit=-1):
    p_list = os.listdir(CODENET_PATH)
    p_qualified_list = []
    p_total = len(p_list)
    p_cur = 0
    langs_set = set()
    for p_id in p_list[:limit]:
        # print(p_id+": ", end="")
        p_cur += 1
        p_folder_path = os.path.join(CODENET_PATH, p_id)
        p_langs = os.listdir(p_folder_path)
        if not set(p_langs).issuperset(set(langs_need)):
            continue
        p_qualified = True
        for lang in langs_need:
            lang_path = os.path.join(p_folder_path, lang)
            codes = os.listdir(lang_path)
            if len(codes) < MIN_SIZE:
                # print("Lack code in {}, codes num: {}".format(lang, len(codes)))
                p_qualified = False
                break
            codes_qualified = 0
            for code_name in codes:
                code_path = os.path.join(lang_path, code_name)
                with open(code_path) as f:
                    code_content = f.read()
                code_content = code_preprocess(code_content)
                code_len = len(code_content)
                if code_len_min < code_len < code_len_max:
                    codes_qualified += 1
                    if codes_qualified == MIN_SIZE:
                        break
            if codes_qualified < MIN_SIZE:
                # print("Lack code in {}, qualified codes num: {}, total: {}".format(lang, codes_qualified, len(codes)))
                p_qualified = False
                break
        if p_qualified:
            print("check: {}/{} {} Qualified.".format(p_cur, p_total, p_id))
            p_qualified_list.append(p_id)
            langs_set.update(p_langs)
            if len(p_qualified_list) == q_need:
                break
    # print("check: {}/{}".format(p_cur, p_total))
    print("Num: "+str(q_need))
    print(p_qualified_list)
    print(langs_set)

def extract_code(lang, p_list, code_len_min, code_len_max):
    p_code_list = []
    for p_id in p_list:
        code_folder_path = os.path.join(CODENET_PATH, p_id, lang)
        if not os.path.exists(code_folder_path):
            continue
        codes = os.listdir(code_folder_path)
        code_list = []
        for code_name in codes:
            code_path = os.path.join(code_folder_path, code_name)
            with open(code_path) as f:
                code_content = f.read()
            code_content = code_preprocess(code_content)
            code_len = len(code_content)
            if code_len_min < code_len < code_len_max:
                code_list.append((code_name, code_content))
        p_code_list.append(code_list)
    return p_code_list

def get_clear_folder(lang):
    if os.path.exists(lang):
        shutil.rmtree(lang)
    os.mkdir(lang)
    return lang

def gen_pair(dataset_list, repeat_check_set, data_jsonl_list, repeat_data_set, code1, code2, flag):
    pair = "\t".join([code1[0], code2[0], flag])
    if pair in repeat_check_set:
        return False
    dataset_list.append(pair)
    repeat_check_set.add(pair)
    pair = "\t".join([code2[0], code1[0], flag])
    repeat_check_set.add(pair)
    if code1[0] not in repeat_data_set:
        data_jsonl_list.append({"func": code1[1], "idx": code1[0]})
        repeat_data_set.add(code1[0])
    if code2[0] not in repeat_data_set:
        data_jsonl_list.append({"func": code2[1], "idx": code2[0]})
        repeat_data_set.add(code2[0])
    return True

def gen_data_jsonl(data_jsonl_list, folder, targetDataJSON="data.jsonl"):
    jsonl_path = os.path.join(folder, targetDataJSON)
    with open(jsonl_path, 'w') as f:
        for item in data_jsonl_list:
            f.write(json.dumps(item) + "\n")

def gen_size_txt(lang, p_list, code_len_min, code_len_max, max_repeat=200, max_pair=200000):
    """
    :max_repeat: max time single code can be trained
    :max_pair: max training set size
    """
    print("Generate pair for {}".format(lang))
    p_code_list = extract_code(lang, p_list, code_len_min, code_len_max)
    dataset_list = []
    repeat_check_set = set()
    data_jsonl_list = []
    repeat_data_set = set()
    folder = get_clear_folder(lang)
    p_num = len(p_code_list)
    max_pair = max_pair // p_num
    max_repeat = max_repeat//2
    for i in range(p_num):
        random.shuffle(p_code_list[i])
    for i in range(p_num):
        c_num = len(p_code_list[i])
        pair_num = 0
        for idx, code in enumerate(p_code_list[i]):
            if pair_num >= max_pair:
                 break
            if idx % 100 == 0:
                print("Problem: {}/{}, Code: {}/{}".format(i+1, p_num, idx+1, c_num))
            for n in range(max_repeat):
                repeat_num = 0
                while True:
                    if repeat_num == 5:
                        break
                    repeat_num += 1
                    p_code = random.choice(p_code_list[i])
                    if p_code is not code and gen_pair(dataset_list, repeat_check_set, data_jsonl_list, repeat_data_set, code, p_code, "1"):
                        pair_num += 1
                        break
            for n in range(max_repeat):
                repeat_num = 0
                while True:
                    if repeat_num == 5:
                        break
                    repeat_num += 1
                    n_p = random.choice(p_code_list)
                    if n_p is not p_code_list[i]:
                        if len(n_p) == 0:
                            continue
                        n_code = random.choice(n_p)
                        if gen_pair(dataset_list, repeat_check_set, data_jsonl_list, repeat_data_set, code, n_code, "0"):
                            pair_num += 1
                            break
    gen_data_jsonl(data_jsonl_list, folder)
    size_txt = "total_{}".format(len(dataset_list))
    with open(os.path.join(folder, size_txt+".txt"), 'w') as f:
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

def check_example(lang, name, map_name='data.jsonl'):
    url_to_code = get_url2code(lang, map_name)
    with open(os.path.join(lang, name)) as f:
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

def split_txt(lang, source, target1, size, target2=""):
    with open(os.path.join(lang, "{}.txt".format(source))) as sf:
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

def gen_new_lang(lang, p_list, code_len_min, code_len_max, max_repeat, max_pair):
    size_txt = gen_size_txt(lang, p_list, code_len_min, code_len_max, max_repeat, max_pair)
    train_txt = split_txt(lang, source=size_txt, target1="temp", size=1400, target2="train")
    split_txt(lang, source="temp_1400", target1="dev", size=400, target2="test")
    os.system("rm {}/temp_1400.txt".format(lang))
    os.system("mv {0}/{1} {0}/train.txt".format(lang, train_txt))

def gen_train(lang, size):
    split_txt(lang=lang, source="train", target1="train", size=size)

def get_data_list(level=-1, lang=""):
    cmd = "tree"
    if level != -1:
        cmd += " -L {}".format(level)
    if lang != "":
        cmd += " {}".format(lang)
    os.system(cmd)

def process_BCBs():
    p_list = []
    n_list = []
    data_jsonl_list = []
    dataset_list = []
    url_to_code = {}
    with open("BCBs/data.jsonl") as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            url_to_code[js['idx']] = code_preprocess(js['func'])
    with open("BCBs/train.txt") as f:
        for line in f:
            ls = line.split()
            i1 = ls[0]
            i2 = ls[1]
            label = ls[2]
            if i1 not in url_to_code or i2 not in url_to_code:
                print(1)
                continue
            if label == "1":
                p_list.append(line)
            else:
                n_list.append(line)
    m_len = min(len(p_list), len(n_list))
    if len(p_list) > m_len:
        p_list = random.sample(p_list, m_len)
    if len(n_list) > m_len:
        n_list = random.sample(n_list, m_len)
    p_list.extend(n_list)
    for line in p_list:
        dataset_list.append(line)
    for idx, code in url_to_code.items():
        data_jsonl_list.append({"func": code, "idx": idx})
    folder = "BCBs"
    gen_data_jsonl(data_jsonl_list, folder)
    size_txt = "total_{}.txt".format(len(dataset_list))
    with open(os.path.join(folder, size_txt), 'w') as f:
        random.shuffle(dataset_list)
        f.write("".join(dataset_list))
    print("data.jsonl: {}, {}".format(len(data_jsonl_list), size_txt))
    lang = "BCBs"
    train_txt = split_txt(lang, source=size_txt, target1="temp", size=1400, target2="train")
    split_txt(lang, source="temp_1400.txt", target1="dev", size=400, target2="test")
    os.system("rm {}/temp_1400.txt".format(lang))
    os.system("mv {0}/{1} {0}/train.txt".format(lang, train_txt))


def get_api2doc(lang, apiFile):
    api_to_doc = {}
    with open(os.path.join(lang, apiFile)) as f:
        for line in f:
            line = line.strip().split()
            api_to_doc[line[0]] = line[0] + " " + " ".join(line[1:])
    return api_to_doc


def addApiInfo(code, api_to_doc):
    for api, doc in api_to_doc.items():
        code = code.replace(api, doc)
    return code


def addApiInfoForList(lang, sourceNameList, targetDataJSON="data_api.jsonl", apiFile="api2doc.txt"):
    url_to_code = get_url2code(lang)
    api_to_doc = get_api2doc(lang, apiFile)

    for sourceName in sourceNameList:
        with open(os.path.join(lang, sourceName+".txt"), 'r') as f:
            for line in f:
                line = line.strip().split()
                url_to_code[line[0]] = addApiInfo(url_to_code[line[0]], api_to_doc)
                url_to_code[line[1]] = addApiInfo(url_to_code[line[1]], api_to_doc)

    data_jsonl_list = []
    for idx, code in url_to_code.items():
        data_jsonl_list.append({"func": code, "idx": idx})
    gen_data_jsonl(data_jsonl_list, "Java", targetDataJSON)

def get_remove_symbol(lang, apiFile):
    sym_list = []
    with open(os.path.join(lang, apiFile)) as f:
        for line in f:
            sym_list.append(line.strip())
    return sym_list


def symbol_process(code, sym_list):
    for symbol in sym_list:
        code = code.replace(symbol, " ")
        code = " ".join(code.split())
    return code


def remove_symbol_for_list(lang, source_name_list, target_data_JSON="data_wo_symbol.jsonl", file="remove_symbol.txt"):
    url_to_code = get_url2code(lang)
    sym_list = get_remove_symbol(lang, file)

    for source_name in source_name_list:
        with open(os.path.join(lang, source_name+".txt"), 'r') as f:
            for line in f:
                line = line.strip().split()
                url_to_code[line[0]] = symbol_process(url_to_code[line[0]], sym_list)
                url_to_code[line[1]] = symbol_process(url_to_code[line[1]], sym_list)

    data_jsonl_list = []
    for idx, code in url_to_code.items():
        data_jsonl_list.append({"func": code, "idx": idx})
    gen_data_jsonl(data_jsonl_list, "Java", target_data_JSON)

if __name__ == '__main__':
    MIN_SIZE = 690
    langs_need = ["Java", "Python", "C++"]
    q_need = 10
    code_len_min = 300
    code_len_max = 600
    # find_qualified_problem(MIN_SIZE, langs_need, q_need, code_len_min, code_len_max)

    # p_list = ['p02696', 'p02743', 'p03086', 'p02646', 'p02659', 'p02953', 'p03471', 'p02572', 'p02658', 'p03001']
    # langs = ['Scheme', 'Objective-C', 'Swift', 'Racket', 'Clojure', 'F#', 'Perl', 'Java', 'Octave', 'Vim', 'Python',
    #          'PHP', 'JavaScript', 'COBOL', 'Elixir', 'Sed', 'MoonScript', 'C', 'Fortran', 'OCaml', 'Rust', 'Dart',
    #          'Visual Basic', 'Haxe', 'Julia', 'Lisp', 'dc', 'bc', 'C#', 'Awk', 'TypeScript', 'Haskell', 'Scala', 'Text',
    #          'Kotlin', 'Lua', 'Erlang', 'Standard ML', 'Bf', 'Prolog', 'Crystal', 'Nim', 'Ruby', 'D', 'Pascal', 'Forth',
    #          'Go', 'C++', 'Cython', 'Bash']

    # for size in [300]:
    #     gen_train(lang="Go", size=size)
    # process_BCBs()
    # check_trainrepeat_pnrate(lang, name="train_5000.txt")
    # get_data_list(level=1, lang="SC")
    # addApiInfoForList("Java", ["train_100", "dev_400", "test_1000"])
    remove_symbol_for_list("Java", ["train_100", "dev_400", "test_1000"])
    check_example(lang="Java", name="train_100.txt", map_name="data_wo_symbol.jsonl")