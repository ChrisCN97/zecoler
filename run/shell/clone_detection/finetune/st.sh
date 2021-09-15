save_dir=./save0802
data_folder=/mnt/sda/cn/python/Ptuning/dataset/new
lang=Python

batch_size=10
lr=3e-5
epoch_num=20
rate=1
eval_step=100

python run.py \
    --output_dir=${save_dir} \
    --data_folder=${data_folder}/${lang} \
    --predictions_name=pre_${lang}.txt \
    --do_eval \
    --do_test \
    --model_type=roberta \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=roberta-base \
    --epoch ${epoch_num} \
    --block_size 400 \
    --train_batch_size ${batch_size} \
    --eval_batch_size 32 \
    --learning_rate ${lr} \
    --max_grad_norm 1.0 \
    --train_data_rate ${rate} \
    --seed 123456 2>&1| tee ${save_dir}/test.log \
&& \
python ../evaluator/evaluator.py -a ${data_folder}//${lang}/test.txt -p ${save_dir}/pre_${lang}.txt -o ${save_dir}/res_${lang}.txt
