1. install dependencies 
```
pip install -r requirements.txt
```

2. To reproduce all results in the paper, run following scripts to get corresponding results:
```
bash scripts/ETTm2.sh
bash scripts/ETTm1.sh
bash scripts/weather.sh
bash scripts/electricity.sh
bash scripts/solar.sh
bash scripts/traffic.sh
```

3. To get result on single set such as etth1 of our model, you can run the following scripts:
```
python -u run_longExp.py  --is_training 1 --model_id test --data 'ETTm1' --data_path 'ETTm1.csv' --pred_len 720 --n_heads 4 --d_model 16 --d_ff 128 --e_layer 1 --features 'M' --fc_dropout 0.8 --dropout 0.8 --pct_start 0.4 --batch_size 128 --model MTETST --patience 3 --seq_len 960 --period_list 96
```