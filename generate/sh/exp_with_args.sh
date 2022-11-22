WORKDIR="/mnt/sda/cn/codet5"
export PYTHONPATH=$WORKDIR

TASK=${1}
SUB_TASK=${2}
MODEL_TAG=${3}
GPU=${4}
DATA_NUM=${5}
BS=${6}
LR=${7}
SRC_LEN=${8}
TRG_LEN=${9}
PATIENCE=${10}
EPOCH=${11}
WARMUP=${12}
MODEL_DIR=${13}
SUMMARY_DIR=${14}
RES_FN=${15}
train_lang=${16}
test_lang=${17}
prompt_num=${18}
need_train=${19}
freeze=${20}
add_prefix=${21}
ctl=${22}
cts=${23}

if [[ $DATA_NUM == -1 ]]; then
  DATA_TAG='all'
else
  DATA_TAG=$DATA_NUM
fi

if [[ ${TASK} == 'multi_task' ]]; then
  FULL_MODEL_TAG=${MODEL_TAG}_${DATA_TAG}_lr${LR}_s${16}
else
  FULL_MODEL_TAG=${MODEL_TAG}_${DATA_TAG}_e${EPOCH}_bs${BS}_tl${train_lang}_p${prompt_num}_f${freeze}
fi


if [[ ${SUB_TASK} == none ]]; then
  OUTPUT_DIR=${MODEL_DIR}/${TASK}/${FULL_MODEL_TAG}
else
  OUTPUT_DIR=${MODEL_DIR}/${TASK}/${SUB_TASK}/${FULL_MODEL_TAG}
fi

CACHE_DIR=${OUTPUT_DIR}/cache_data
RES_DIR=${OUTPUT_DIR}/prediction
LOG=${OUTPUT_DIR}/train.log
mkdir -p ${OUTPUT_DIR}
mkdir -p ${CACHE_DIR}
mkdir -p ${RES_DIR}

if [[ $MODEL_TAG == roberta ]]; then
  MODEL_TYPE=roberta
  TOKENIZER=roberta-base
  MODEL_PATH=roberta-base
elif [[ $MODEL_TAG == roberta-large ]]; then
  MODEL_TYPE=roberta
  TOKENIZER=roberta-large
  MODEL_PATH=roberta-large
elif [[ $MODEL_TAG == codebert ]]; then
  MODEL_TYPE=roberta
  TOKENIZER=roberta-base
  MODEL_PATH=microsoft/codebert-base
elif [[ $MODEL_TAG == codeberta ]]; then
  MODEL_TYPE=roberta
  TOKENIZER=huggingface/CodeBERTa-small-v1
  MODEL_PATH=huggingface/CodeBERTa-small-v1
elif [[ $MODEL_TAG == bart_base ]]; then
  MODEL_TYPE=bart
  TOKENIZER=facebook/bart-base
  MODEL_PATH=facebook/bart-base
elif [[ $MODEL_TAG == codet5_small ]]; then
  MODEL_TYPE=codet5
  TOKENIZER=Salesforce/codet5-small
  MODEL_PATH=Salesforce/codet5-small
elif [[ $MODEL_TAG == codet5_base ]]; then
  MODEL_TYPE=codet5
  TOKENIZER=Salesforce/codet5-base
  MODEL_PATH=Salesforce/codet5-base
elif [[ $MODEL_TAG == codet5_large ]]; then
  MODEL_TYPE=codet5
  TOKENIZER=Salesforce/codet5-large
  MODEL_PATH=Salesforce/codet5-large
elif [[ $MODEL_TAG == test_mlm ]]; then
  MODEL_TYPE=roberta
  TOKENIZER=roberta-base
  MODEL_PATH='../pretrain/test-mlm'
elif [[ $MODEL_TAG == codebert-with-lang-v1 ]]; then
  MODEL_TYPE=roberta
  TOKENIZER=roberta-base
  MODEL_PATH='../pretrain/codebert-with-lang-v1'
elif [[ $MODEL_TAG == codet5-with-lang-v1 ]]; then
  MODEL_TYPE=codet5
  TOKENIZER=Salesforce/codet5-base
  MODEL_PATH='../pretrain/codet5-with-lang-v1'
fi


if [[ ${TASK} == 'multi_task' ]]; then
  RUN_FN=${WORKDIR}/run_multi_gen.py
  MULTI_TASK_AUG='--max_steps '${16}' --save_steps '${17}' --log_steps '${18}
elif [[ ${TASK} == 'clone' ]]; then
  RUN_FN=${WORKDIR}/run_clone.py
elif [[ ${TASK} == 'defect' ]] && [[ ${MODEL_TYPE} == 'roberta' ||  ${MODEL_TYPE} == 'bart' ]]; then
  RUN_FN=${WORKDIR}/run_defect.py
else
  RUN_FN=${WORKDIR}/run_gen.py
fi

if [[ ${need_train} == 'True' ]]; then
  CUDA_VISIBLE_DEVICES=${GPU} \
    python ${RUN_FN}  ${MULTI_TASK_AUG}   \
    --do_train --do_eval --do_test  \
    --task ${TASK} --sub_task ${SUB_TASK} --model_type ${MODEL_TYPE} --data_num ${DATA_NUM}  \
    --num_train_epochs ${EPOCH} --warmup_steps ${WARMUP} --learning_rate ${LR}e-5 --patience ${PATIENCE} \
    --tokenizer_name=${TOKENIZER}  --model_name_or_path=${MODEL_PATH} --data_dir ${WORKDIR}/data  \
    --cache_path ${CACHE_DIR}  --output_dir ${OUTPUT_DIR}  --summary_dir ${SUMMARY_DIR} \
    --save_last_checkpoints --always_save_model --res_dir ${RES_DIR} --res_fn ${RES_FN} \
    --train_batch_size ${BS} --eval_batch_size ${BS} --max_source_length ${SRC_LEN} --max_target_length ${TRG_LEN} \
    --train_lang ${train_lang} --test_lang ${test_lang} --prompt_num ${prompt_num} --freeze ${freeze} \
    --add_prefix ${add_prefix} --model_tag ${MODEL_TAG} --continue_train_lang ${ctl} --continue_train_size ${cts} \
    2>&1 | tee ${LOG}
else
  CUDA_VISIBLE_DEVICES=${GPU} \
    python ${RUN_FN}  ${MULTI_TASK_AUG}   \
    --do_test  \
    --task ${TASK} --sub_task ${SUB_TASK} --model_type ${MODEL_TYPE} --data_num ${DATA_NUM}  \
    --num_train_epochs ${EPOCH} --warmup_steps ${WARMUP} --learning_rate ${LR}e-5 --patience ${PATIENCE} \
    --tokenizer_name=${TOKENIZER}  --model_name_or_path=${MODEL_PATH} --data_dir ${WORKDIR}/data  \
    --cache_path ${CACHE_DIR}  --output_dir ${OUTPUT_DIR}  --summary_dir ${SUMMARY_DIR} \
    --save_last_checkpoints --always_save_model --res_dir ${RES_DIR} --res_fn ${RES_FN} \
    --train_batch_size ${BS} --eval_batch_size ${BS} --max_source_length ${SRC_LEN} --max_target_length ${TRG_LEN} \
    --train_lang ${train_lang} --test_lang ${test_lang} --prompt_num ${prompt_num} --freeze ${freeze} --add_prefix ${add_prefix} \
    2>&1 | tee ${LOG}
fi
