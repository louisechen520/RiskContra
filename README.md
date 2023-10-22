# RiskContra
RiskContra: A Contrastive Approach to Forecast Traffic Risks with Multi-Kernel Networks

## Introduction
Spatial-temporal learning is the mainstream approach to exploring complex evolving patterns. However, two intrinsic challenges lie in traffic accident forecasting, preventing the straightforward adoption of spatial-temporal learning. First, the temporal observations of traffic accidents exhibit ultra-rareness due to the inherent properties of accident occurrences, which leads to the severe scarcity of risk samples in learning accident patterns. Second, the spatial distribution of accidents is severely imbalanced from region to region, which poses a serious challenge to forecast the spatially diversified risks. To tackle the above challenges, we propose RiskContra, a Contrastive learning approach with multi-kernel networks, to forecast the Risk of traffic accidents. Specifically, to address the first challenge (i.e. temporal rareness), we design a novel contrastive learning approach, which leverages the periodic patterns to derive a tailored mixup strategy for risk sample augmentation. 
% that employs a customized mixup strategy to generate augmented risk samples. This way, the contrastively learned features can better represent the risk samples, thus capturing higher-quality accident patterns for forecasting. To address the second challenge (i.e. spatial imbalance), we design the multi-kernel networks to capture the hierarchical correlations from multiple spatial granularities. This way, disparate regions can utilize the multi-granularity correlations to enhance the forecasting performance across regions. 

## Framework
![image](https://github.com/chenchl19941118/RiskContra/assets/25497533/b7bb1659-804e-418f-975d-46bcac5c667b)

## Result

## Visualization
![image](https://github.com/chenchl19941118/RiskContra/assets/25497533/4221b987-4597-4d00-9c74-c3a4d9d055fa)

## Usage

train model on NYC:
```
python train.py --config config/nyc/RiskContra_NYC_Config.json --gpus 0
```


train model on Chicago:
```
python train.py --config config/chicago/RiskContra_Chicago_Config.json --gpus 0
```




