if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=1440
model_name=MTETST

root_path_name=./dataset/
data_path_name=solar_AL.txt
model_id_name=Solar
data_name=Solar
exp_name=solar

random_seed=2021
for pred_len in 96 192 336 720
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --period_list 144 \
      --enc_in 137 \
      --e_layers 1 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.5\
      --fc_dropout 0.5\
      --head_dropout 0\
      --des 'Exp' \
      --train_epochs 100\
      --patience 3\
      --lradj 'TST'\
      --pct_start 0.4\
      --itr 1 --batch_size 32 --learning_rate 0.0001 >logs/LongForecasting/$exp_name'_'$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done