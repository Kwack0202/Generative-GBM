# Generative-GBM
Development of a Trading System Integrating Financial Engineering Models and Artificial Intelligence

## Abstract
This study proposes a scenario-based trading system that integrates generative models the GBM framework. The core idea is to use the GBM framework to model the uncertainty of stock price variations and to optimize the wiener process by employing a Latent Diffusion Model to depart from the log-normal distribution assumption of the standard GBM. This approach demonstrates that by introducing generative models, the terms of the wiener process can be reparameterized by learning the unique distribution and time-varying structure of market data. This provides realistic virtual price paths and extends the dataset for stock prediction to scenario sets, contributing to the optimization of investment strategies. The generated scenarios are selected through risk assessment and learn the patterns of stock variations using a Transformer encoder. The empirical analysis conducted on 150 stocks representing 10 sectors of the S&P 500 index showed that LDM outperformed existing generative models in terms of original data mimicking, with a win rate of 0.567 and a cumulative return of 38.434% in backtesting results. Additionally, the proposed trading system demonstrated consistent adaptability across various risk profiles based on growth and volatility, as well as different market conditions such as bear and bull markets. Consequently, the proposed trading system has proven its potential to support investors in achieving profitable returns by leveraging scenarios that combine realism and diversity.

## Concept of proposed trading system
<p align="center">
  <img src="./asset/Fig2.png" width="100%">
</p>

### Geometric Brownian Motion
<p align="center">
  <img src="./asset/Fig1.png" width="80%">
</p>

<div align="center" style="font-size: 180%;">

$$
\Large S_t = S_0 e^{(\mu - \frac{\sigma^2}{2})t + \sigma W_t}
$$

</div>

- **Idea** : GBM models the future path of stock prices as a stochastic process, systematically representing market uncertainty. Each scenario represents a possible path of asset returns, and the entire set provides insights into the overall direction and variation of stock price uncertainty.

### Latent Diffusion Model
<p align="center">
  <img src="./asset/Fig3.png" width="100%">
</p>

## 🛠 System
- **CPU** `AMD Ryzen 9 5950X 16-Core Processor`
- **GPU** `NVIDIA GeForce RTX 4080`
- **Memory RAM** `128GB`

**The computational efficiency of Generative-GBM is proportional to the CPU's power(Logical processor)**

## 📑 Usage
### Requirments
- **python version** `3.8`
- **TA Library** `TA_Lib-0.4.24-cp38-cp38-win_amd64.whl`
- **Other packages** `Packages in common_imports.py`

### run.py for data preparing & backtesting
- To run the system, the parser arguments must be passed using the `run.py` and `sh files`
- The `sh file` is divided into subfolders and multiple steps within the ./scripts/ folder.

### The scripts folder structure is as follows:
```
./scripts/
├── 01_origin_data/
│     └──data_download.sh
│
├── 02_gbm/
│     └──GemerativeAI_gbm
│     │   ├──train_Diffusion.sh
│     │   ├──train_GAN.sh
│     │   └──   :
│     │
│     └──Methematicl_gbm
│         └──   :
│     
├── 03_stock_prediction/
│     ├──stock_prediction.sh
│     ├──backtesting.sh
│     └──   :
```

#### Example command (Git Bash)
```
sh ./scripts/01_origin_data/data_download.sh
```
#### Example code in sh file
```
#!/bin/bash
python run.py \
    --task_name data_download \
    --metadata ./datasets/metadata_info.csv \
    --exp_root_path SP500 \
    --start_day 2019-01-01 \
    --end_day 2025-01-01 \
```

## 📊 Generative-GBM Scenario result 
<p align="center">
  <img src="./asset/Fig5.png" width="100%">
</p>

## 📈 Backtesting 📉
### Treemap
<p align="center">
  <img src="./asset/Fig8.png" width="100%">
</p>

### Trading plot
<p align="center">
  <img src="./asset/Fig9.png" width="100%">
</p>

### XAI (Scenario Activation Rate)
<p align="center">
  <img src="./asset/Fig10.png" width="100%">
</p>