echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash ft_bertbase.sh TASK_NAME MODEL_NAME EPOCH SEED"
echo "for example: bash scripts/ft_bertbase.sh CoLA bert-base-uncased 10 42"
echo "=============================================================================================================="

TASK_NAME=$1
MODEL_NAME=$2
EPOCH=$3
SEED=$4

# User change USER_ROOT:
USER_ROOT=
PROJECT_DIR=${USER_ROOT}/DAP-BERT
GLUE_DIR=${PROJECT_DIR}/glue_data

# export PYTORCH_PRETRAINED_BERT_CACHE=${PROJECT_DIR}/models/.pytorch_pretrained_bert

python ${PROJECT_DIR}/fine_tune.py \
  --task_name ${TASK_NAME} \
  --do_train  \
  --do_eval  \
  --do_lower_case  \
  --data_dir ${GLUE_DIR}/${TASK_NAME}  \
  --cache_dir ~/lyu2002_1tb/.cache  \
  --bert_model ${MODEL_NAME}  \
  --max_seq_length 128 \
  --train_batch_size 128  \
  --learning_rate 2e-5  \
  --num_train_epochs ${EPOCH}  \
  --output_dir ${PROJECT_DIR}/models/bert-base-uncased-ft/${TASK_NAME} \
  --test_output_dir ${PROJECT_DIR}/test_output/${TASK_NAME} \
  --seed ${SEED} \
  --gradient_accumulation_steps 4
  
