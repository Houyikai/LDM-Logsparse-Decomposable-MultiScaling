## background
This is a method for time series forecasting. Short fixed-length inputs are the main bottleneck of deep learning methods in long time-series forecasting tasks. Prolonging input length causes overfitting, rapidly deteriorating accuracy(left pic). Our method can adapt to longer context, thereby achieving better performance(right pic).
![image](https://github.com/Houyikai/MTE/assets/39182537/b2a37717-cda2-44d3-a0a5-adef22f95c6e)

## model
![image](https://github.com/Houyikai/MTE/assets/39182537/4165815c-97f9-46f8-a635-787a5bf4e5c2)
The dynamic patterns of time series are often dominated by several frequency components. We found that handling different frequency components of time series separately can improve prediction performance and efficiently utilize larger contexts. First, Fourier analysis is used to determine the periods (1/freq) of the top-k amplitudes in the series. Then, the series is decomposed into a set of periodic sequences based on these period lengths, with each sequence dominated by a specific length of periodic pattern. Each periodic sequence is then embedded and modeled using its period length as the patch size (token size) through a Periodic pattern recognition (PPR) module, allowing the model to focus on different scales of the series.

## why it can accomdate long inputs 
We first need to understand the cause of overfitting, which is:

**A single, fixed patch size causes the model to focus only on the temporal patterns at a primary scale of the sequence (e.g., 24 points, or one day), thus not requiring a long context (512). **

Therefore, it is necessary to introduce multi-resolution attention (i.e., MTE) and shorten the context of smaller resolutions to avoid overfitting. This is a straightforward implementation,  there are many better methods for multi-resolution attention, refer to the computer vision field. However, we believe sequence decoupling is essential.

## why WIDE than DEEP
This method employs a wide, low-coupling architecture instead of a deep network, which intuitively seems suboptimal. However, some facts about current forecasting models are: (1) the number of layers is usually small (1~3), and (2) the encoder layers between the input and output layers have a relatively small impact on the prediction results (~10%). These facts compel us to consider adopting a wide structure and making improvements at the embedding layer and the prediction layer rather than the slightly-involved encoding layer(input-output layer).

## Performance
主要结果，mae 和 mse， 数值越低越好
![image](https://github.com/Houyikai/MTE/assets/39182537/81e89266-7adc-45a3-ac5e-033bf1b6c8e6)

可视化结果
![Snipaste_2024-07-20_20-57-49](https://github.com/user-attachments/assets/c64fb174-8cfa-4219-8cfa-e6b0e69bc6f4)


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
