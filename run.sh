# PRE_TRAN="hfl/chinese-macbert-large"
PRE_TRAN="bert-base-chinese"

# python src/main.py \
# --train_path="$HOME/Datasets/train_dataset_v2.tsv" \
# --test_path="$HOME/Datasets/test_dataset_v2.tsv" \
# --model_save_dir="$HOME/opt/tiger/SC_task/model" \
# --fc1_dim=256 \
# --fc2_dim=32 \
# --dropout=0.1 \
# --learning_rate=0.0003 \
# --emotions_type=6 \
# --batch_size=2 \
# --have_up \
# --eval_step=3000 \
# --fix_length=500 \
# --hidden_size=300 \
# --fix_length_lstm=100 \
# --pre_train_name=$PRE_TRAN \
# > log/train.log 2>&1 &



python src/main.py \
--train_path="$HOME/Datasets/train_dataset_v2.tsv" \
--test_path="$HOME/Datasets/test_dataset_v2.tsv" \
--model_save_dir="$HOME/opt/tiger/SC_task/model/$MODE_SUB" \
--fc1_dim=256 \
--fc2_dim=32 \
--dropout=0.1 \
--learning_rate=0.0003 \
--eval_step=800 \
--emotions_type=505 \
--have_up \
--fix_length=500 \
--batch_size=8 \
--pre_train_name=$PRE_TRAN \
> log/train.log 2>&1 &