save_dir=./save_old
lang=Java
batch_size=10
lr=5e-5

python run.py \
    --output_dir=${save_dir} \
    --data_folder=../dataset/${lang} \
    --predictions_name=pre_${lang}.txt \
    --do_train \
    --train_batch_size ${batch_size} \
    --model_type=roberta \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=roberta-base \
    --epoch 2 \
    --block_size 400 \
    --eval_batch_size 32 \
    --learning_rate ${lr} \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee ${save_dir}/train.log \
&& \
python run.py \
    --output_dir=${save_dir} \
    --data_folder=../dataset/${lang} \
    --predictions_name=pre_${lang}.txt \
    --do_eval \
    --do_test \
    --model_type=roberta \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=roberta-base \
    --epoch 2 \
    --block_size 400 \
    --train_batch_size ${batch_size} \
    --eval_batch_size 32 \
    --learning_rate ${lr} \
    --max_grad_norm 1.0 \
    --seed 123456 2>&1| tee ${save_dir}/test.log \
&& \
python ../evaluator/evaluator.py -a ../dataset/${lang}/test.txt -p ${save_dir}/pre_${lang}.txt -o ${save_dir}/res_${lang}.txt

nohup python run.py \
    --output_dir=./save0629 \
    --data_folder=../dataset/Java \
    --predictions_name=pre_Java.txt \
    --do_train \
    --train_batch_size 10 \
    --model_type=roberta \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=roberta-base \
    --epoch 2 \
    --block_size 400 \
    --eval_batch_size 32 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee ./save0629/train.log \
&& \
python run.py \
    --output_dir=./save_test2 \
    --data_folder=../dataset/test2 \
    --predictions_name=pre_java.txt \
    --do_eval \
    --do_test \
    --model_type=roberta \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=roberta-base \
    --epoch 2 \
    --block_size 400 \
    --train_batch_size 10 \
    --eval_batch_size 32 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --seed 123456 2>&1| tee ./save_test2/test.log \
&& \
python ../evaluator/evaluator.py -a ../dataset/test2/test.txt -p ./save_test2/pre_java.txt -o ./save_test2/res_java.txt \
> extract.log 2>&1 &

nohup ./s.sh > save0802_f/s.log 2>&1 &
19554

python run.py \
    --output_dir=./save_source \
    --data_folder=../dataset/source \
    --predictions_name=pre_java.txt \
    --do_train \
    --train_batch_size 10 \
    --model_type=roberta \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=roberta-base \
    --epoch 2 \
    --block_size 400 \
    --eval_batch_size 32 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee ./save_source/train_java.log \
&& \
python run.py \
    --output_dir=./save_source \
    --data_folder=../dataset/source \
    --predictions_name=pre_java.txt \
    --do_test \
    --model_type=roberta \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=roberta-base \
    --epoch 2 \
    --block_size 400 \
    --train_batch_size 10 \
    --eval_batch_size 32 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --seed 123456 2>&1| tee ./save_source/test_java.log \
&& \
python ../evaluator/evaluator.py -a ../dataset/source/test.txt -p ./save_source/pre_java.txt -o ./save_source/res_java.txt \
&& \
python run.py \
    --output_dir=./save_source \
    --data_folder=../dataset/source2 \
    --predictions_name=pre_source2.txt \
    --do_eval \
    --do_test \
    --model_type=roberta \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=roberta-base \
    --epoch 2 \
    --block_size 400 \
    --train_batch_size 10 \
    --eval_batch_size 32 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --seed 123456 2>&1| tee ./save_source/test_source2.log \
&& \
python ../evaluator/evaluator.py -a ../dataset/C#/test.txt -p ./save_source/pre_c#.txt -o ./save_source/res_c#.txt \
&& \
python run.py \
    --output_dir=./save \
    --data_folder=../dataset/java2 \
    --predictions_name=pre_java2.txt \
    --do_eval \
    --do_test \
    --model_type=roberta \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=roberta-base \
    --epoch 2 \
    --block_size 400 \
    --train_batch_size 10 \
    --eval_batch_size 32 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --seed 123456 2>&1| tee ./save/test_java2.log \
&& \
python ../evaluator/evaluator.py -a ../dataset/java2/test.txt -p ./save/pre_java2.txt -o ./save/res_java2.txt