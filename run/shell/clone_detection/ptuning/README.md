option parameter
--freeze_plm

help
lab1405: 
conda activate ptuning
cd /mnt/sda/cn/zeroshot/run/shell/clone_detection/ptuning
/mnt/sda/cn/zeroshot/run/output/clone_detection/ptuning/log
lab1405new: 
conda activate allennlp

nohup ./train_test.sh > ../../../output/clone_detection/ptuning/log/java_train_32_test.log 2>&1 &
./train_test.sh 2>&1 | tee ../../../output/clone_detection/ptuning/log/java_train_32_test.log 