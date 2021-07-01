echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash scripts/demo.sh FLOP_RATIO ALPHA_UPDATE_RATIO TASK EPOCH_S EPOCH_FT SEED"
echo "for example: bash scripts/demo.sh 0.5 0.1 CoLA 10 10 42"
echo "=============================================================================================================="
FLOP_RATIO=$1
ALPHA_UR=$2
TASK_NAME=$3
EPOCH_S=$4
EPOCH_FT=$5
SEED=$6
FLOP_WEIGHT=1

# User change USER_ROOT absolute path:
USER_ROOT=
PROJECT_DIR=${USER_ROOT}/DAP-BERT

TEACHER=TinyBERT_4L_312D/${TASK_NAME}
STUDENT=TinyBERT_4L_312D/${TASK_NAME}
# TEACHER=bert-base-uncased-ft/${TASK_NAME}
# STUDENT=bert-base-uncased-ft/${TASK_NAME}
OUT=tinybert-r${FLOP_RATIO}/${TASK_NAME}
GLUE_DIR=${PROJECT_DIR}/glue_data/${TASK_NAME}

TEACHER_DIR=${PROJECT_DIR}/models/${TEACHER}
STUDENT_DIR=${PROJECT_DIR}/models/${STUDENT}
INT_DIST_DIR=${PROJECT_DIR}/models/${OUT}
TEST_OUT_DIR=${PROJECT_DIR}/test_result/${TASK_NAME}
PRED_DIST_DIR=${INT_DIST_DIR}_search

python ${PROJECT_DIR}/search.py \
			--pred_distill \
			--teacher_model ${TEACHER_DIR} \
			--student_model ${STUDENT_DIR} \
			--data_dir ${GLUE_DIR} \
			--task_name ${TASK_NAME} \
			--output_dir ${PRED_DIST_DIR} \
			--max_seq_length 128 \
			--train_batch_size 32 \
			--num_train_epochs ${EPOCH_S} \
			--FLOP_weight ${FLOP_WEIGHT} \
			--FLOP_ratio ${FLOP_RATIO} \
			--do_lower_case \
			--arch_flop_loss \
			--FLOP_tolerant 0.01 \
			--search_heads \
			--search_ff \
			--store_alpha \
			--mul_prog_arch_cstr \
			--arch_update_ratio ${ALPHA_UR} \
			--force_save_model \
			# --aug_train \
			--seed ${SEED}

FT_INT_DIST_DIR=${PROJECT_DIR}/models/${OUT}_ft

python ${PROJECT_DIR}/fine_tune_pruned.py \
			--int_distill \
			--teacher_model ${TEACHER_DIR} \
			--student_model ${STUDENT_DIR} \
			--searched_model ${PRED_DIST_DIR} \
			--data_dir ${GLUE_DIR} \
			--task_name ${TASK_NAME} \
			--output_dir ${FT_INT_DIST_DIR} \
			--max_seq_length 128 \
			--train_batch_size 32 \
			--num_train_epochs ${EPOCH_FT} \
			--do_lower_case \
			--search_heads \
			--search_ff \
			--seed ${SEED}

FT_PRED_DIST_DIR=${FT_INT_DIST_DIR}_pred

python ${PROJECT_DIR}/fine_tune_pruned.py \
			--pred_distill \
			--teacher_model ${TEACHER_DIR} \
			--student_model ${FT_INT_DIST_DIR} \
			--searched_model ${PRED_DIST_DIR} \
			--data_dir ${GLUE_DIR} \
			--task_name ${TASK_NAME} \
			--output_dir ${FT_PRED_DIST_DIR} \
			--max_seq_length 128 \
			--train_batch_size 32 \
			--num_train_epochs ${EPOCH_FT} \
			--do_lower_case \
			--search_heads \
			--search_ff \
			--seed ${SEED}

EVAL_OUT_DIR=${FT_PRED_DIST_DIR}/eval_${TASK_NAME}

python ${PROJECT_DIR}/fine_tune_pruned.py \
			--do_eval \
			--teacher_model ${TEACHER_DIR} \
			--student_model ${FT_PRED_DIST_DIR} \
			--searched_model ${PRED_DIST_DIR} \
			--data_dir ${GLUE_DIR} \
			--task_name ${TASK_NAME} \
			--output_dir ${EVAL_OUT_DIR} \
			--max_seq_length 128 \
			--train_batch_size 32 \
			--num_train_epochs ${EPOCH_FT} \
			--do_lower_case \
			--search_heads \
			--search_ff \
			--seed ${SEED}

python ${PROJECT_DIR}/fine_tune_pruned.py \
			--do_test \
			--teacher_model ${TEACHER_DIR} \
			--student_model ${FT_PRED_DIST_DIR} \
			--searched_model ${PRED_DIST_DIR} \
			--data_dir ${GLUE_DIR} \
			--task_name ${TASK_NAME} \
			--output_dir ${TEST_OUT_DIR} \
			--max_seq_length 128 \
			--train_batch_size 32 \
			--num_train_epochs ${EPOCH_FT} \
			--do_lower_case \
			--search_heads \
			--search_ff \
			--seed ${SEED}