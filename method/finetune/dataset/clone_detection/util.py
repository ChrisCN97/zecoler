import json
import os
import random

def get_url2code(dataset, name='data.jsonl'):
    url_to_code = {}
    with open(os.path.join(dataset, name)) as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            url_to_code[js['idx']] = js['func']
    return url_to_code

def sample(dataset, name):
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