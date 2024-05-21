if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=1680
model_name=MTETST

root_path_name=./dataset/
data_path_name=ETTh2.csv
model_id_name=MTETST
data_name=ETTh2

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
      --period_list 24,168 \
      --enc_in 7 \
      --e_layers 1 \
      --n_heads 4 \
      --d_model 16 \
      --d_ff 128 \
      --dropout 0.8 \
      --pct_start 0.4 \
      --fc_dropout 0.8 \
      --head_dropout 0 \
      --des 'Exp' \
      --patience 3 \
      --train_epochs 100 \
      --itr 1 --batch_size 128 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done