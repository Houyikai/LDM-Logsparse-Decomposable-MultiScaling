## background
Short fixed-length inputs are the main bottleneck of deep learning methods in long time-series forecasting tasks. Prolonging input length causes overfitting, rapidly deteriorating accuracy(left pic). Our method can adapt to longer context, thereby achieving better performance(right pic).
![image](https://github.com/Houyikai/MTE/assets/39182537/b2a37717-cda2-44d3-a0a5-adef22f95c6e)

## model
![image](https://github.com/Houyikai/MTE/assets/39182537/4165815c-97f9-46f8-a635-787a5bf4e5c2)
This is a method for time series forecasting. First, Fourier analysis is used to determine the periods (1/freq) of the top-k amplitudes in the series. Then, the series is decomposed into a set of periodic sequences based on these period lengths, with each sequence dominated by a specific length of periodic pattern. Each periodic sequence is then embedded and modeled using its period length as the patch size (token size) through a Periodic pattern recognition (PPR) module, allowing the model to focus on different scales of the series.

## why it can accomdate long inputs 
We first need to understand the cause of overfitting, which is:

**A single, fixed patch size causes the model to focus only on the temporal patterns at a primary scale of the sequence (e.g., 24 points, or one day), thus not requiring a long context (512). **

Therefore, it is necessary to introduce multi-resolution attention (i.e., MTE) and shorten the context of smaller resolutions to avoid overfitting. This is a straightforward implementation,  there are many better methods for multi-resolution attention, refer to the computer vision field. However, we believe sequence decoupling is essential.

## how to use
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
