

python src/main.py \
--train_path="$HOME/Datasets/train_dataset_v2.tsv" \
--test_path="$HOME/Datasets/test_dataset_v2.tsv" \
--model_save_dir="$HOME/opt/tiger/SC_task/model" \
--fc1_dim=256 \
--fc2_dim=32 \
--dropout=0.1 \
--learning_rate=0.0003 \
--emotions_type=0 \
--batch_size=8 \
--have_up \
--eval_step=1000 \
--fix_length=50 \
> log/train.log 2>&1 &