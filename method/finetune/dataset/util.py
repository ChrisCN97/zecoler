import json
import os
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

def gen_data_jsonl(data_jsonl_list, folder):
    jsonl_path = os.path.join(folder, "data.jsonl")
    with open(jsonl_path, 'w') as f:
        for item in data_jsonl_list:
            f.write(json.dumps(item) + "\n")

def trans_file(source, target, url_to_code, data_jsonl_list):
    with open(source) as f:
        with open(target, 'w') as f2:
            for line in f:
                lines = line.split()
                data_jsonl_list.append({"func": url_to_code[lines[0]], "idx": lines[0]})
                data_jsonl_list.append({"func": url_to_code[lines[1]], "idx": lines[1]})
                f2.write(line)

def gen_dataset(lang, size):
    lang_folder = os.path.join(TARGET_PATH, lang)
    folder = get_clear_folder(os.path.join(lang_folder, size))
    url_to_code = get_url2code(lang)
    data_jsonl_list = []
    source = os.path.join(SOURCE_PATH, lang, "{}{}.txt".format(TRAIN_PREFIX, size))
    if not os.path.exists(source):
        print("{}/{}{}.txt needs to be created!".format(lang, TRAIN_PREFIX, size))
        return
    trans_file(source, os.path.join(folder, "train.txt"), url_to_code, data_jsonl_list)
    trans_file(os.path.join(SOURCE_PATH, lang, DEV),
               os.path.join(folder, "valid.txt"), url_to_code, data_jsonl_list)
    trans_file(os.path.join(SOURCE_PATH, lang, TEST),
               os.path.join(folder, "test.txt"), url_to_code, data_jsonl_list)
    gen_data_jsonl(data_jsonl_list, folder)

def gen_test():
    folder = os.path.join(TARGET_PATH, "Java", "test")
    os.makedirs(folder)
    lang = "Java"
    size = 32
    url_to_code = get_url2code(lang)
    data_jsonl_list = []
    source = os.path.join(SOURCE_PATH, lang, "{}{}.txt".format(TRAIN_PREFIX, size))
    if not os.path.exists(source):
        print("{}/{}{}.txt needs to be created!".format(lang, TRAIN_PREFIX, size))
        return
    trans_file(source, os.path.join(folder, "train.txt"), url_to_code, data_jsonl_list)
    os.system("cp {0}/train.txt {0}/valid.txt".format(folder))
    os.system("cp {0}/train.txt {0}/test.txt".format(folder))
    gen_data_jsonl(data_jsonl_list, folder)

def get_data_list(level=-1, lang=""):
    cmd = "tree"
    if level != -1:
        cmd += " -L {}".format(level)
    if lang != "":
        cmd += " {}".format(lang)
    os.system(cmd)



if __name__ == '__main__':
    task_list = ["clone_detection", "code_search", "name_predict"]
    TARGET_PATH = task_list[0]
    SOURCE_PATH = os.path.join(SOURCE_PATH, TARGET_PATH)

    # for task in task_list:
    #     TARGET_PATH = task
    #     for size in [5000]:
    #         gen_dataset(lang="Go", size=str(size))
    gen_dataset(lang="Go", size=str(5000))
