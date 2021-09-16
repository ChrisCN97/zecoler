import matplotlib.pyplot as plt
import numpy as np
import json
import os

def plot_loss(loss_list_name):
    loss_list = np.load(loss_list_name)
    print(len(loss_list))
    plt.figure()
    plt.plot(np.arange(len(loss_list)), loss_list)
    plt.title(loss_list_name)
    plt.show()

def get_url2code(dataset, name='data.jsonl'):
    url_to_code = {}
    with open(os.path.join(dataset, name)) as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            url_to_code[js['idx']] = js['func']
    return url_to_code

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
