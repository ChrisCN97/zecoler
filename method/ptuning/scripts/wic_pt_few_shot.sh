export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

PATH_TO_DATA_DIR=../../FewGLUE_32dev
PATH_TO_OUTPUT_DIR=../output/wic_0719

python3 ../cli.py \
--data_dir ${PATH_TO_DATA_DIR}/WiC \
--model_type albert \
--model_name_or_path albert-xxlarge-v2 \
--task_name wic \
--output_dir ${PATH_TO_OUTPUT_DIR}/wic \
--do_eval \
--do_train \
--pet_per_gpu_eval_batch_size 16 \
--pet_per_gpu_train_batch_size 2 \
--pet_gradient_accumulation_steps 1 \
--pet_max_seq_length 256 \
--pet_max_steps 100 \
--pattern_ids 1 \
--pet_repetitions 1 \
--eval_every_step 20
