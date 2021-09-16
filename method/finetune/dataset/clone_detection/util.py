import json
import os
import random
import shutil

SOURCE_PATH = "../../../../dataset/clone_detection"
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
    os.mkdir(folder)
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
    if not os.path.exists(lang):
        os.mkdir(lang)
    folder = get_clear_folder(os.path.join(lang, size))
    url_to_code = get_url2code(lang)
    data_jsonl_list = []
    trans_file(os.path.join(SOURCE_PATH, lang, "{}{}.txt".format(TRAIN_PREFIX, size)),
               os.path.join(folder, "train.txt"), url_to_code, data_jsonl_list)
    trans_file(os.path.join(SOURCE_PATH, lang, DEV),
               os.path.join(folder, "valid.txt"), url_to_code, data_jsonl_list)
    trans_file(os.path.join(SOURCE_PATH, lang, TEST),
               os.path.join(folder, "test.txt"), url_to_code, data_jsonl_list)
    gen_data_jsonl(data_jsonl_list, folder)

if __name__ == '__main__':
    gen_dataset(lang="Java", size="32")