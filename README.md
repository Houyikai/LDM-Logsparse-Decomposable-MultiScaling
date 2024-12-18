## background
This is a method for time series forecasting. Short fixed-length inputs are the main bottleneck of deep learning methods in long time-series forecasting tasks. Prolonging input length causes overfitting, rapidly deteriorating accuracy(left pic). Our method can adapt to longer context, thereby achieving better performance(right pic).
<img src="https://github.com/user-attachments/assets/2d67d52d-152d-4c4a-896d-3f9c47e6fb84" width="600" />

## model
<img src="https://github.com/user-attachments/assets/e2439ede-1500-415c-9fda-3c65de95f94a" width="600" />

The multiscale modeling approach, exemplified by TimeMixer, has shown promise in modeling long-term dependencies, especially in real-world phenomena like traffic with multiple temporal patterns. TimeMixer uses two main modules: the Past-Decomposable-Mixing (PDM), which mixes seasonal and trend components at different scales, and the Future-Multipredictor-Mixing (FMM), which combines multiple forecasters for better accuracy.

However, the sampling approach has three main limitations: 1) **Insufficient context**: Shorter inputs lead to prediction errors as the model lacks sufficient context for effective learning. 2) **Non-stationarity**: Downsampling introduces additional non-stationary components, increasing complexity. 3) **Limited applicability**: Multiscale downsampling requires specialized modules (like PDM) for aggregation, adding overhead and reducing scalability.

## why it can accomdate long inputs 
We first need to understand the cause of overfitting, which is:

**A single, fixed patch size causes the model to focus only on the temporal patterns at a primary scale of the sequence (e.g., 24 points, or one day), thus not requiring a long context (512). **

Therefore, it is necessary to introduce multi-resolution analysis and shorten the context of smaller resolutions to avoid overfitting. This is a straightforward implementation,  there are many better methods such as multi-resolution attention.

## why WIDE than DEEP
This method employs a wide, low-coupling architecture instead of a deep network, which intuitively seems suboptimal. However, some facts about current forecasting models are: (1) the number of layers is usually small (1~3), and (2) the encoder layers between the input and output layers have a relatively small impact on the prediction results (~10%). These facts compel us to consider adopting a wide structure and making improvements at the embedding layer and the prediction layer rather than the slightly-involved encoding layer(input-output layer).

## Performance
The main results, MAE (Mean Absolute Error) and MSE (Mean Squared Error), with lower values indicating better predictive performance.
Multivariate Benchmark
![Snipaste_2024-12-18_19-43-52](https://github.com/user-attachments/assets/aab844aa-dec5-48d1-9772-c5ae84284564)
Univariate Benchmark
<img src="https://github.com/user-attachments/assets/714f56f5-27a3-415b-afe7-67a3160b9126" width="600" />
Performance promotion to old baseline 
<img src="https://github.com/user-attachments/assets/a1c7ab2b-1284-4fc1-8c69-cb0843557b8d" width="600" />
Visualization results.
![Snipaste_2024-12-18_20-11-34](https://github.com/user-attachments/assets/0b1d8b90-6d1f-4a40-a505-1ec49285b04c)


## how to use
1. install dependencies 
```
pip install -r requirements.txt
```

2. prepare data
You can obtain the well pre-processed datasets from [[Google Drive]](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing) or [[Baidu Drive]](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy), Then place the downloaded data in the folder`./dataset`. thanks to [[thuml]](https://github.com/thuml)


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
