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

PROJUECT="$HOME/opt/tiger/SC_task"

python src/main.py \
--predict \
--test_path="$HOME/Datasets/test_dataset.tsv" \
--batch_size=4 \
--model_0="$HOME/opt/tiger/SC_task/model/model5/505_2021_10_13_15_52_KFID_0_0.6891.pkl" \
--model_1="$HOME/opt/tiger/SC_task/model/model5/505_2021_10_13_15_52_KFID_0_KFID_1_0.6939.pkl" \
--model_2="$HOME/opt/tiger/SC_task/model/model5/505_2021_10_13_15_52_KFID_0_KFID_1_KFID_2_0.7027.pkl" \
--model_3="$HOME/opt/tiger/SC_task/model/model5/505_2021_10_13_15_52_KFID_0_KFID_1_KFID_2_KFID_3_0.6761.pkl" \
--model_4="$HOME/opt/tiger/SC_task/model/model5/505_2021_10_13_15_52_KFID_0_KFID_1_KFID_2_KFID_3_KFID_4_0.6932.pkl" \
--model_5="$HOME/opt/tiger/SC_task/model/model5/505_2021_10_13_15_52_KFID_0_KFID_1_KFID_2_KFID_3_KFID_4_KFID_5_0.6780.pkl" \
--model_6="$HOME/opt/tiger/SC_task/model/model5/505_2021_10_13_15_52_KFID_0_KFID_1_KFID_2_KFID_3_KFID_4_KFID_5_KFID_6_0.6859.pkl" \
--model_7="$HOME/opt/tiger/SC_task/model/model5/505_2021_10_13_15_52_KFID_0_KFID_1_KFID_2_KFID_3_KFID_4_KFID_5_KFID_6_KFID_7_0.6699.pkl" \
--model_8="$HOME/opt/tiger/SC_task/model/505_2021_10_14_12_31_KFID_0_0.6804.pkl" \
--model_9="$HOME/opt/tiger/SC_task/model/505_2021_10_14_14_03_KFID_0_0.6878.pkl" \
--model_all="$HOME/opt/tiger/SC_task/model/model8/506_2021_10_15_20_34_0.6827.pkl" \
--predict_save_path="$HOME/opt/tiger/SC_task/output/ans_13.tsv" \
--fix_length=500 \
--fix_length_lstm=100 \
> log/predict.log 2>&1 &
