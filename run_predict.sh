#!/bin/sh

python src/main.py \
--predict \
--test_path="$HOME/Datasets/test_dataset.tsv" \
--batch_size=8 \
--model_0="$HOME/opt/tiger/SC_task/model/0_2021_10_06_19_11_0.pkl" \
--model_1="$HOME/opt/tiger/SC_task/model/1_2021_10_06_19_11_1.pkl" \
--model_2="$HOME/opt/tiger/SC_task/model/2_2021_10_06_19_11_1.pkl" \
--model_3="$HOME/opt/tiger/SC_task/model/3_2021_10_06_19_11_2.pkl" \
--model_4="$HOME/opt/tiger/SC_task/model/4_2021_10_06_19_11_4.pkl" \
--model_5="$HOME/opt/tiger/SC_task/model/5_2021_10_06_19_10_3.pkl" \
--predict_save_path="$HOME/opt/tiger/SC_task/output/ans_1.tsv" \
> log/predict.log 2>&1 &