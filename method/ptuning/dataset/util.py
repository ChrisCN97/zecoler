import os
import json
import random
import shutil

SOURCE_PATH = "../../../dataset"
TARGET_PATH = ""
TRAIN_PREFIX = "train_"
DEV = "dev_400.txt"
TEST = "test_1000.txt"

def get_url2code(lang, name='data.jsonl'):
    url_to_code = {}
    with open(os.path.join(SOURCE_PATH, lang, name)) as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            url_to_code[js['idx']] = js['func']
    return url_to_code

def get_clear_folder(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
    return folder

def change_format(source, target, url_to_code, idx):
    jsonl_list = []
    with open(source) as f:
        for line in f:
            line = line.split()
            code1 = url_to_code[line[0]]
            code2 = url_to_code[line[1]]
            label = True if line[2] == "1" else False
            jsonl_list.append({
                "code1": code1,
                "code2": code2,
                "idx": idx,
                "label": label,
            })
            idx += 1
    with open(target, 'w') as f:
        random.shuffle(jsonl_list)
        for item in jsonl_list:
            f.write(json.dumps(item) + "\n")
    return idx

def gen_dataset(lang, size, suffix="", dataJSON=""):
    size = str(size)
    lang_folder = os.path.join(TARGET_PATH, lang)
    folder = get_clear_folder(os.path.join(lang_folder, size) + suffix)
    if dataJSON == "":
        url_to_code = get_url2code(lang)
    else:
        url_to_code = get_url2code(lang, dataJSON)
    idx = 0
    source = os.path.join(SOURCE_PATH, lang, "{}{}.txt".format(TRAIN_PREFIX, size))
    if not os.path.exists(source):
        print("{}/{}{}.txt needs to be created!".format(lang, TRAIN_PREFIX, size))
        return
    idx = change_format(source,
                        os.path.join(folder, "train.jsonl"), url_to_code, idx)
    idx = change_format(os.path.join(SOURCE_PATH, lang, DEV),
                        os.path.join(folder, "dev32.jsonl"), url_to_code, idx)
    change_format(os.path.join(SOURCE_PATH, lang, TEST),
                  os.path.join(folder, "val.jsonl"), url_to_code, idx)

def get_data_list(level=-1, lang=""):
    cmd = "tree"
    if level != -1:
        cmd += " -L {}".format(level)
    if lang != "":
        cmd += " {}".format(lang)
    os.system(cmd)

def gen_test():
    folder = os.path.join(TARGET_PATH, "Java", "test")
    if not os.path.exists(folder):
        os.makedirs(folder)
    lang = "Java"
    size = 32
    url_to_code = get_url2code(lang)
    source = os.path.join(SOURCE_PATH, lang, "{}{}.txt".format(TRAIN_PREFIX, size))
    if not os.path.exists(source):
        print("{}/{}{}.txt needs to be created!".format(lang, TRAIN_PREFIX, size))
        return
    change_format(source, os.path.join(folder, "train.jsonl"), url_to_code, 0)
    os.system("cp {0}/train.jsonl {0}/val.jsonl".format(folder))
    os.system("cp {0}/train.jsonl {0}/dev32.jsonl".format(folder))

if __name__ == "__main__":
    task_list = ["clone_detection", "code_search", "name_predict"]
    TARGET_PATH = task_list[0]
    SOURCE_PATH = os.path.join(SOURCE_PATH, TARGET_PATH)

    # for task in task_list:
    #     TARGET_PATH = task
    #     for size in [500]:
    #         gen_dataset(lang="Go", size=str(size))
    # gen_dataset(lang="Go", size=str(5000))
    # gen_dataset(lang="Java", size=str(100))
    gen_dataset(lang="Java", size=str(100), suffix="_wosym", dataJSON="data_wo_symbol.jsonl")