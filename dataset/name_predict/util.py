import json
import os
import random
import re
import shutil
from tqdm import tqdm


def gen_dataset(lang, folder, min_code_len, max_code_len):
    dataset_list = []
    data_jsonl_list = []
    pair_list = []
    idx = 1
    with open("CSN_source/source/{}.jsonl".format(lang)) as f:
        for line in tqdm(f.readlines()):
            line = line.strip()
            js = json.loads(line)
            code = code_preprocess(js['code'], lang)
            doc = txt_preprocess(js['func_name'])
            if len(code) > 0 and len(doc) > 0 and min_code_len < len(code) + len(doc) < max_code_len:
                pair_list.append((idx, idx + 1))
                dataset_list.append("\t".join([str(idx), str(idx + 1), "1"]))
                data_jsonl_list.append({"func": doc, "idx": str(idx)})
                data_jsonl_list.append({"func": code, "idx": str(idx + 1)})
                idx += 2
    for doc_i, code_i in pair_list:
        while True:
            d2, c2 = random.choice(pair_list)
            if c2 != code_i:
                break
        dataset_list.append("\t".join([str(doc_i), str(c2), "0"]))
    get_clear_folder(folder)
    gen_data_jsonl(data_jsonl_list, folder)
    size_txt = "total_{}.txt".format(len(dataset_list))
    with open(os.path.join(folder, size_txt), 'w') as f:
        random.shuffle(dataset_list)
        f.write("\n".join(dataset_list))
    print("data.jsonl: {}, {}".format(len(data_jsonl_list), size_txt))
    lang = folder
    train_txt = split_txt(lang, source=size_txt, target1="temp", size=1400, target2="train")
    split_txt(lang, source="temp_1400.txt", target1="dev", size=400, target2="test")
    os.system("rm {}/temp_1400.txt".format(lang))
    os.system("mv {0}/{1} {0}/train.txt".format(lang, train_txt))


def txt_preprocess(txt):
    return " ".join(txt.split())


# java python javascript go
def code_preprocess(code, lang):
    if lang == "python":
        return python_preprocess(code)
    return java_preprocess(code)


def python_preprocess(code):
    code = re.sub(r"#.*", '', code)
    code = re.sub(r"\"\"\".*\"\"\"", '', code, flags=re.S)
    code = txt_preprocess(code)
    return code


def java_preprocess(code):
    code = re.sub(r"//.*", '', code)
    code = re.sub(r"/\*.*\*/", '', code, flags=re.S)
    code = txt_preprocess(code)
    return code


def get_clear_folder(lang):
    if os.path.exists(lang):
        shutil.rmtree(lang)
    os.makedirs(lang)
    return lang


def gen_data_jsonl(data_jsonl_list, folder, targetDataJSON="data.jsonl"):
    jsonl_path = os.path.join(folder, targetDataJSON)
    with open(jsonl_path, 'w') as f:
        for item in data_jsonl_list:
            f.write(json.dumps(item) + "\n")


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
            idx = len(sl) - 1
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

def gen_train(lang, size):
    split_txt(lang=lang, source="train.txt", target1="train", size=size)

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
        print("Match!")
    else:
        print("Not Match!")

def get_url2code(lang, name='data.jsonl'):
    url_to_code = {}
    with open(os.path.join(lang, name)) as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            url_to_code[js['idx']] = js['func']
    return url_to_code

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
                url_to_code[line[1]] = symbol_process(url_to_code[line[1]], sym_list)

    data_jsonl_list = []
    for idx, code in url_to_code.items():
        data_jsonl_list.append({"func": code, "idx": idx})
    gen_data_jsonl(data_jsonl_list, "Java", target_data_JSON)

if __name__ == "__main__":
    # check_example(lang="Java", name="train.txt")
    # for lang, folder in [("python", "Python"), ("go", "Go"), ("javascript", "JavaScript")]:
    #     gen_dataset(lang, folder, min_code_len=800, max_code_len=1000)
    # for size in [5000]:
    #     gen_train(lang="Go", size=size)
    # gen_dataset(lang="javascript", folder="JavaScript", min_code_len=700, max_code_len=900)
    # addApiInfoForList("Java", ["train_100", "dev_400", "test_1000"])
    # remove_symbol_for_list("Java", ["train_100", "dev_400", "test_1000"])
    check_example(lang="Java", name="train_100.txt")  # , map_name="data_wo_symbol.jsonl"