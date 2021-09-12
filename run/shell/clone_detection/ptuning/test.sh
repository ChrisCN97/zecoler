export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

PROJECT_NAME=clonedet_0802_f
DATASET_NAME=JavaScript

step=6000
eval_step=200
batch=2
show=0

python3 ../cli.py \
--pattern_ids 10 \
--freeze_plm \
--show_limit ${show} \
--data_dir ../../CloneDetection_32dev/${DATASET_NAME} \
--model_type roberta \
--model_name_or_path microsoft/codebert-base \
--task_name clonedet \
--output_dir ../output/${PROJECT_NAME} \
--do_eval \
--pet_per_gpu_eval_batch_size 16 \
--pet_per_gpu_train_batch_size ${batch} \
--pet_gradient_accumulation_steps 1 \
--pet_max_seq_length 512 \
--pet_max_steps ${step} \
--pet_repetitions 1 \
--eval_every_step ${eval_step} \
--overwrite_output_dir \
--embed_size 768 \
--show_limit 0 | tee log/${PROJECT_NAME}_print.log