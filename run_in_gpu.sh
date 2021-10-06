#!/bin/sh
source ~/.bashrc
source activate telma
TYPE=0

python src/main.py \
--train_path="$HOME/Datasets/train_dataset_v2.tsv" \
--test_path="$HOME/Datasets/test_dataset_v2.tsv" \
--model_save_dir="$HOME/opt/tiger/SC_task/model" \
--fc1_dim=256 \
--fc2_dim=32 \
--dropout=0.5 \
--learning_rate=0.00003 \
--emotions_type=$TYPE \
--batch_size=12 \
# > log/train.log 2>&1 &