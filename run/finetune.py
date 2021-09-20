from util import get_dataset
from server import S1, S2
import os

langs = ["Java", "Python", "JavaScript", "PHP", "Ruby", "Go", "C#", "C++", "C", "Haskell", "Kotlin", "Fortran"]

def finetune(
        model_type="roberta",
        config_name="microsoft/codebert-base",
        model_name_or_path="microsoft/codebert-base",
        tokenizer_name="roberta-base",
        task_name="clone_detection",
        lang="Java",
        size=32,
        output="Java_32",
        do_train=True,
        freeze_plm=False,
        train_batch=10,
        epoch=20,
        train_batch_size=10,
        eval_step=100,
        learning_rate=1e-5,
        do_test=True,
        env=S2,
        is_part=False,
        nohup=False
):
    cmd = "python ../method/finetune/code/run.py " \
          "--block_size 400 " \
          "--eval_batch_size 32 " \
          "--max_grad_norm 1.0 " \
          "--evaluate_during_training " \
          "--train_data_rate 1 " \
          "--seed 123456 "
    cmd += "--model_type {} ".format(model_type)
    cmd += "--config_name {} ".format(config_name)
    cmd += "--model_name_or_path {} ".format(model_name_or_path)
    cmd += "--tokenizer_name {} ".format(tokenizer_name)
    data_dir = "../method/finetune/dataset/{}/{}/{}".format(task_name, lang, size)
    if not os.path.exists(data_dir):
        if env != S2:
            print("Get dataset {}/{}/{} from server2".format(task_name, lang, size))
            get_dataset(method="finetune", task=task_name, lang=lang, size=size, from_server=S2)
        else:
            print("Dataset {}/{}/{} needs to be generated!".format(task_name, lang, size))
    cmd += "--data_folder {} ".format(data_dir)
    log_path = "output/{}/finetune/log/{}".format(task_name, output)
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    log = "{}/{}.log".format(log_path, lang)
    output = "output/{}/finetune/{}".format(task_name, output)
    if do_test and \
            (not os.path.exists(output) or not os.listdir(output)):
        print("Lack of {} for zero shot".format(output))
    cmd += "--output_dir {} ".format(output)
    if freeze_plm:
        cmd += "--freeze_plm "
    cmd += "--train_batch_size {} ".format(train_batch)
    cmd += "--epoch {} ".format(epoch)
    cmd += "--train_batch_size {} ".format(train_batch_size)
    cmd += "--learning_rate {} ".format(learning_rate)
    if do_train:
        cmd += "--do_train "
        cmd += "--save_steps {} ".format(eval_step)
    if do_test:
        cmd += "--do_eval "
        cmd += "--do_test "
        cmd += "--predictions_name pre_{}.txt".format(lang)
        cmd += "\npython ../method/finetune/evaluator/evaluator.py "
        cmd += "-a {}/test.txt ".format(data_dir)
        cmd += "-p {}/pre_{}.txt ".format(output, lang)
        cmd += "-o {}".format(log)
    if do_train:
        print(output)
    if is_part:
        return cmd
    if nohup:
        cmd = "nohup {} > output/{}/finetune/log/single.log 2>&1 &".format(cmd, task_name)
    else:
        cmd = "{} 2>&1 | tee output/{}/finetune/log/single.log".format(cmd, task_name)
    print(cmd)
    with open("run.sh", 'w') as f:
        f.write(cmd)

def finetune_clone_detection_list(task_dicts, env, check_data=False):
    cmd = ""
    pre_time = 0
    for task in task_dicts:
        c = finetune(
            lang=task["lang"],
            size=task["size"],
            output=task["output"],
            do_train=task["do_train"],
            freeze_plm=task["freeze_plm"],
            epoch=task["epoch"],
            eval_step=task["eval_step"],
            do_test=task["do_test"],
            env=env,
            is_part=True)
        cmd += c + "\n"
        if task["do_train"]:
            pre_time += int(task["epoch"])*int(task["size"])/10*0.011
        else:
            pre_time += 0.633
    print(cmd)
    if not check_data:
        with open("run.sh", 'w') as f:
            f.write(cmd)
    if env == S1:
        print("conda activate ptuning")
    print("nohup ./run.sh > output/clone_detection/finetune/log/task_list.log 2>&1 &".format(cmd))
    h, m = divmod(pre_time, 60)
    print("%dh %02dmin" % (h, m))

if __name__ == "__main__":
    # task_dicts = [{"lang": "Java", "size": 5000, "output": "Java_5000", "do_train": True, "freeze_plm": False,
    #                "epoch": 8, "eval_step": 100, "do_test": False}]
    task_dicts = []
    for item in[(10000,200), (7000,140), (3000,100), (1000,100)]:
        task_dicts.append({"lang": "Java", "size": item[0], "output": "Java_{}".format(item[0]), "do_train": True,
                           "freeze_plm": False, "epoch": 20, "eval_step": item[1], "do_test": False})
        for t_lang in langs:
            task_dicts.append({"lang": t_lang, "size": 32, "output": "Java_{}".format(item[0]), "do_train": False,
                               "freeze_plm": False, "epoch": 8, "eval_step": 100, "do_test": True})
    finetune_clone_detection_list(task_dicts, S2, check_data=False)

