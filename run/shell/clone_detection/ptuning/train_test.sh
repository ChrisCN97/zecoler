export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

LANG=Java
SIZE=5000
FOLDER=${LANG}_${SIZE}
step=10000
eval_step=100
batch=5
show=0

python3 ../../../../method/ptuning/cli.py \
--pattern_ids 10 \
--show_limit ${show} \
--data_dir ../../../../method/ptuning/dataset/clone_detection/${LANG}/${SIZE} \
--model_type roberta \
--model_name_or_path microsoft/codebert-base \
--task_name clonedet \
--output_dir ../../../output/clone_detection/ptuning/${FOLDER} \
--do_eval \
--do_train \
--pet_per_gpu_eval_batch_size 16 \
--pet_per_gpu_train_batch_size ${batch} \
--pet_gradient_accumulation_steps 1 \
--pet_max_seq_length 512 \
--pet_max_steps ${step} \
--pet_repetitions 1 \
--eval_every_step ${eval_step} \
--overwrite_output_dir \
--embed_size 768 \
--show_limit 0