import os
import json

ROOT = "../../../../dataset/clone_detection"

def get_url2code(dataset, name='data.jsonl'):
    url_to_code = {}
    with open(os.path.join(dataset, name)) as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            url_to_code[js['idx']] = js['func']
    return url_to_code