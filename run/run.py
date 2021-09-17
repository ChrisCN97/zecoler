import os
import shutil
from server import S1, S2, USER, IP
import matplotlib.pyplot as plt
import numpy as np

def get_clear_folder(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
    return folder

def scp_get(source, target, port, user=USER, ip=IP):
    os.system("scp -r -P {} {}@{}:{} {}".format(port, user, ip, source, target))

def get_dataset(method, task, lang, size, from_server):
    source = os.path.join(from_server["root"], "method", method, "dataset", task, lang, str(size))
    target = os.path.join("../method", method, "dataset", task, lang)
    if not os.path.exists(target):
        os.makedirs(target)
    scp_get(source, target, from_server["port"])

def plot_loss(output, folder, name):
    # name: acc.npy / loss.npy
    loss_list = np.load(os.path.join("../../../output/clone_detection/ptuning", output, folder, name))
    print(len(loss_list))
    plt.figure()
    plt.plot(np.arange(len(loss_list)), loss_list)
    plt.title("{}: {}".format(output, name))
    plt.show()

def ptuning(
        model_type="roberta",
        model_name_or_path="microsoft/codebert-base",
        embed_size=768,
        task_name="clone_detection",
        lang="Java",
        size=32,
        output="Java_32",
        do_train=True,
        freeze_plm=False,
        pattern_ids=10,
        train_batch=5,
        max_step=100,
        eval_step=50,
        pet_repetitions=1,
        do_eval=True,
        show_limit=0,
        nohup=False):
    cmd = "python3 ../method/ptuning/cli.py " \
          "--pet_per_gpu_eval_batch_size 16 " \
          "--pet_gradient_accumulation_steps 1 " \
          "--pet_max_seq_length 512 "
    cmd += "--model_type {} ".format(model_type)
    cmd += "--model_name_or_path {} ".format(model_name_or_path)
    cmd += "--embed_size {} ".format(embed_size)
    task_map = {"clone_detection": "clonedet"}
    cmd += "--task_name {} ".format(task_map[task_name])
    data_dir = "../method/ptuning/dataset/{}/{}/{}".format(task_name, lang, size)
    if not os.path.exists(data_dir):
        print("Get dataset {}/{}/{} from server2".format(task_name, lang, size))
        get_dataset(method="ptuning", task=task_name, lang=lang, size=size, from_server=S2)
    cmd += "--data_dir {} ".format(data_dir)
    cmd += "--output_dir output/{}/ptuning/{} ".format(task_name, output)
    if do_train:
        cmd += "--do_train "
        cmd += "--overwrite_output_dir "
    if do_eval:
        cmd += "--do_eval "
    if freeze_plm:
        cmd += "--freeze_plm "
    cmd += "--pattern_ids {} ".format(pattern_ids)
    cmd += "--pet_per_gpu_train_batch_size {} ".format(train_batch)
    cmd += "--pet_max_steps {} ".format(max_step)
    cmd += "--eval_every_step {} ".format(eval_step)
    cmd += "--pet_repetitions {} ".format(pet_repetitions)
    cmd += "--show_limit {}".format(show_limit)
    log = "output/{}/ptuning/log/{}_{}.log".format(task_name, output, lang)
    if nohup:
        cmd = "nohup {} > {} 2>&1 &".format(cmd, log)
    else:
        cmd = "{} 2>&1 | tee {}".format(cmd, log)
    # cmd = "conda activate ptuning && " + cmd
    print(cmd)
    # print("{}/p{}-i0".format(output, pattern_ids))
    # os.system(cmd)
    return cmd

if __name__ == "__main__":
    # get_dataset(method="ptuning", task="clone_detection", lang="Java", size="5000", from_server=S2)
    ptuning(
        model_type="roberta",
        model_name_or_path="microsoft/codebert-base",
        embed_size=768,
        task_name="clone_detection",
        lang="Python",
        size=32,
        output="Java_5000",
        do_train=False,
        freeze_plm=False,
        pattern_ids=10,
        train_batch=5,
        max_step=10,
        eval_step=5,
        pet_repetitions=1,
        do_eval=True,
        show_limit=0,
        nohup=True)