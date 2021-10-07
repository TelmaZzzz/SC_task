#!/bin/sh

python src/main.py \
--predict \
--test_path="$HOME/Datasets/test_dataset.tsv" \
--batch_size=8 \
--model_0="$HOME/opt/tiger/SC_task/model/model1/0_2021_10_07_00_04_0.pkl" \
--model_1="$HOME/opt/tiger/SC_task/model/model1/1_2021_10_07_00_04_0.pkl" \
--model_2="$HOME/opt/tiger/SC_task/model/model1/2_2021_10_07_00_04_0.pkl" \
--model_3="$HOME/opt/tiger/SC_task/model/model1/3_2021_10_07_00_05_6.pkl" \
--model_4="$HOME/opt/tiger/SC_task/model/model1/4_2021_10_07_00_05_1.pkl" \
--model_5="$HOME/opt/tiger/SC_task/model/model1/5_2021_10_07_00_05_3.pkl" \
--predict_save_path="$HOME/opt/tiger/SC_task/output/ans_2.tsv" \
> log/predict.log 2>&1 &