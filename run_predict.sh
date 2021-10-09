#!/bin/sh

# python src/main.py \
# --predict \
# --test_path="$HOME/Datasets/test_dataset.tsv" \
# --batch_size=8 \
# --model_0="$HOME/opt/tiger/SC_task/model/model3/0_2021_10_08_23_04_0.7539.pkl" \
# --model_1="$HOME/opt/tiger/SC_task/model/model3/1_2021_10_08_23_05_0.7062.pkl" \
# --model_2="$HOME/opt/tiger/SC_task/model/model3/2_2021_10_08_23_05_0.7396.pkl" \
# --model_3="$HOME/opt/tiger/SC_task/model/model3/3_2021_10_08_23_05_0.6836.pkl" \
# --model_4="$HOME/opt/tiger/SC_task/model/model3/4_2021_10_08_23_05_0.6592.pkl" \
# --model_5="$HOME/opt/tiger/SC_task/model/model3/5_2021_10_08_23_05_0.5993.pkl" \
# --predict_save_path="$HOME/opt/tiger/SC_task/output/ans_6.tsv" \
# > log/predict_0.log 2>&1 &

python src/main.py \
--predict \
--test_path="$HOME/Datasets/test_dataset.tsv" \
--batch_size=8 \
--model_all="$HOME/opt/tiger/SC_task/model/model4/100_2021_10_09_14_56_0.6810.pkl" \
--predict_save_path="$HOME/opt/tiger/SC_task/output/ans_7.tsv" \
--fix_length=100 \
> log/predict.log 2>&1 &
