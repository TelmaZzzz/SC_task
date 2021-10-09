#!/bin/sh
source ~/.bashrc
source activate telma
TYPE=150
MODE_SUB="model4"

python src/main.py \
--train_path="$HOME/Datasets/train_dataset_v2.tsv" \
--test_path="$HOME/Datasets/test_dataset_v2.tsv" \
--model_save_dir="$HOME/opt/tiger/SC_task/model/$MODE_SUB" \
--fc1_dim=256 \
--fc2_dim=32 \
--dropout=0.1 \
--learning_rate=0.0003 \
--eval_step=200 \
--emotions_type=$TYPE \
--have_up \
--fix_length=150 \
--batch_size=32 \
# > log/train.log 2>&1 &