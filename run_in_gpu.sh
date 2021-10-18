#!/bin/sh
source ~/.bashrc
source activate telma
TYPE=506
MODE_SUB="model8"
# PRE_TRAN="nghuyong/ernie-1.0"
# PRE_TRAN="hfl/chinese-macbert-base"
# PRE_TRAN="hfl/chinese-macbert-large"
# PRE_TRAN="hfl/chinese-roberta-wwm-ext"
PRE_TRAN="bert-base-chinese"
# export CUDA_VISIBLE_DEVICES=0

# python src/main.py \
# --train_path="$HOME/Datasets/train_dataset_v2.tsv" \
# --test_path="$HOME/Datasets/test_dataset_v2.tsv" \
# --model_save_dir="$HOME/opt/tiger/SC_task/model/$MODE_SUB" \
# --fc1_dim=768 \
# --fc2_dim=128 \
# --dropout=0.1 \
# --learning_rate=0.0003 \
# --eval_step=400 \
# --emotions_type=$TYPE \
# --have_up \
# --fix_length=500 \
# --fix_length_lstm=75 \
# --batch_size=16 \
# --pre_train_name=$PRE_TRAN \
# --hidden_size=300 \
# > log/train.log 2>&1 &


# python src/main.py \
# --train_path="$HOME/Datasets/train_dataset_v2.tsv" \
# --test_path="$HOME/Datasets/test_dataset_v2.tsv" \
# --model_save_dir="$HOME/opt/tiger/SC_task/model/$MODE_SUB" \
# --fc1_dim=768 \
# --fc2_dim=128 \
# --dropout=0.1 \
# --learning_rate=0.0003 \
# --eval_step=4000 \
# --emotions_type=$TYPE \
# --have_up \
# --fix_length=500 \
# --batch_size=2 \
# --pre_train_name=$PRE_TRAN \


python src/main.py \
--train_path="$HOME/Datasets/train_dataset_v2.tsv" \
--test_path="$HOME/Datasets/test_dataset_v2.tsv" \
--model_save_dir="$HOME/opt/tiger/SC_task/model/$MODE_SUB" \
--fc1_dim=256 \
--fc2_dim=32 \
--dropout=0.3 \
--learning_rate=0.0003 \
--eval_step=500 \
--emotions_type=$TYPE \
--have_up \
--fix_length=500 \
--batch_size=16 \
--epoch=20 \
--pre_train_name=$PRE_TRAN \