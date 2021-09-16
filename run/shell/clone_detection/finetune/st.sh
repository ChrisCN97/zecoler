save_dir=../../../output/clone_detection/finetune/java_train_400_test
data_folder=../../../../method/finetune/dataset/clone_detection
lang=Java_r
batch_size=10
lr=1e-5
epoch_num=4
rate=1
eval_step=100

python ../../../../method/finetune/code/run.py \
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
python ../../../../method/finetune/evaluator/evaluator.py -a ${data_folder}//${lang}/test.txt -p ${save_dir}/pre_${lang}.txt -o ../../../output/clone_detection/finetune/log/res_${lang}.txt