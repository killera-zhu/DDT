
# DDT: A Dual-Masking Dual-Expert Transformer for Energy Time-Series Forecasting

This repository is the official implementation of the paper: **"DDT: A Dual-Masking Dual-Expert Transformer for Energy Time-Series Forecasting"**.

DDT is a novel and robust deep learning framework designed specifically for high-precision energy time-series forecasting. It addresses critical challenges in multi-source heterogeneous data fusion and resolves the conflict between strict causal consistency and adaptive feature selection.

## 🌟 Key Features

* **Dual-Masking Mechanism**: A synergistic combination of a strict **Causal Mask** (to ensure theoretical causal consistency) and a data-driven **Dynamic Mask** (to adaptively focus on salient historical information).
* **Dual-Expert System**: A "divide and conquer" architecture that decouples modeling into two specialized pathways:
    * **Temporal Expert**: Captures intra-series dynamics using multi-scale dilated convolutions.
    * **Channel Expert**: Models inter-series relationships using an MLP-based variable dependency module.
* **Dynamic Gated Fusion**: Intelligently integrates the outputs of the dual experts.
* **State-of-the-Art Performance**: Achieved Rank-1 performance across **10 challenging energy benchmark datasets** (including ETTh, Electricity, Solar, Traffic, etc.).

## 🏗️ Model Architecture

The overall architecture of DDT consists of a comprehensive data processing pipeline, the Dual-Masking Mechanism, and the Dual-Expert System.

<div align="center">
  <img src="./figures/DDT.png" alt="DDT Model Architecture" width="800">
  <br>
  <em>Figure: Schematic diagram of the DDT model structure.</em>
</div>

## 📂 Datasets

We evaluated DDT on **10 widely-used energy benchmark datasets**:

* **ETTh1, ETTh2, ETTm1, ETTm2**: Electricity transformer temperature data.
* **Electricity**: Hourly electricity consumption of 321 clients.
* **Solar**: Solar power generation records.
* **Wind**: Wind power generation records.
* **Traffic**: Traffic usage data.
* **Weather**: Local climatological data.
* **ILI**: Influenza-like illness data.
* **Exchange**: Exchange rate data.

Data should be placed in the `./dataset/` directory.

## 🚀 Getting Started

### Requirements

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

*Key dependencies:*
* darts==0.25.0
* matplotlib==3.7.5
* numpy==1.24.4
* pandas==1.5.3
* salesforce_merlion==2.0.2
* scikit_learn==1.3.2
* scipy==1.10.1
* statsmodels==0.14.1
* torch==2.4.1
* ray==2.10.0
* tqdm==4.66.4
* dash==2.17.0
* dash-bootstrap-components==1.6.0
* reformer-pytorch==1.4.4
* lightgbm==4.1.0

## 📊 Results

DDT consistently outperforms strong baselines (including Pathformer, iTransformer, TimeMixer, and PatchTST) across various prediction horizons.

<div align="center">
  <img src="./figures/results.png" alt="results" width="800">
  <br>
  <em>Figure: Schematic diagram of the DDT model structure.</em>
</div>

*For detailed statistical evaluation and full results tables, please refer to the experimental section of our paper.*
#