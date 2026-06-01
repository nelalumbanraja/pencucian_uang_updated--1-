# Explainable Edge AI for Cost-Efficient and Trustworthy Anti-Money Laundering Systems

This repository contains the implementation and experimental results for the paper **"Explainable Edge AI for Cost-Efficient and Trustworthy Anti-Money Laundering Systems"**.

The project develops a lightweight Anti-Money Laundering (AML) detection system using machine learning and explainable artificial intelligence under the framework named **XE-AML**. The main objective is to support cost-efficient and trustworthy AML detection by moving transaction screening closer to the transaction source through an Edge AI approach.

The proposed system evaluates three ensemble-based models — Random Forest, XGBoost, and LightGBM — to detect suspicious financial transactions. SHAP is used to explain model predictions and support transparency, auditability, and regulatory review.

In addition to classification performance, this project evaluates computational performance for inference deployment, including inference latency, model size, memory consumption, CPU usage, and transaction throughput.

The original dataset is not included in this repository because the CSV file exceeds GitHub's file size limit.

---

## Project Overview

This project includes:

- Anti-Money Laundering transaction classification using ensemble-based machine learning
- Data preprocessing and feature engineering on the SAML-D dataset
- Class imbalance handling using RandomOverSampler on the training set only
- Model training and evaluation: Random Forest, XGBoost, and LightGBM
- Model evaluation using Accuracy, Precision, Recall, F1-Score, ROC-AUC, and PR-AUC
- SHAP-based explainability for suspicious transaction interpretation
- Inference benchmark for Edge AI computational performance evaluation
- Measurement of model size, memory footprint, CPU usage, latency, and throughput

---

## Proposed System: XE-AML

XE-AML (Explainable Edge AI for Anti-Money Laundering) is designed to support near-real-time suspicious transaction detection by deploying a lightweight machine learning model closer to transaction origination points.

Instead of relying only on centralized data centers, the Edge AI approach allows transaction features to be processed at local or regional edge nodes. This reduces communication latency, lowers infrastructure cost, and supports faster detection before suspicious funds move further through the financial system.

The system consists of three main components:

1. **Transaction Detection Layer**  
   Detects whether a transaction is normal or suspicious using lightweight machine learning.

2. **Lightweight Machine Learning Layer**  
   Uses LightGBM as the main edge model because it provides strong predictive performance with relatively low computational cost, small model size, and efficient memory usage.

3. **Explainable AI Layer**  
   Uses SHAP to explain why a transaction is classified as suspicious or normal, supporting compliance officer review and regulatory audit.

---

## Dataset

This project uses the **SAML-D dataset**, a transaction-level dataset for anti-money laundering detection.

The target variable is:

```text
Is_laundering
  0 = Normal transaction
  1 = Suspicious transaction
```

The dataset is highly imbalanced — suspicious transactions represent only **0.1039%** of the full dataset. The experiment applies subsampling and oversampling strategies to improve model learning on the minority class.

| Dataset Aspect | Description |
|---|---|
| Dataset name | SAML-D |
| Original transactions | 9,504,852 |
| Suspicious transactions | 9,873 (0.1039%) |
| Experimental sample size | 209,873 transactions |
| Final input features | 22 |
| Target variable | Is_laundering |

The original `SAML-D.csv` file is not included in this repository due to GitHub's file size limitation. Place the dataset in the project directory before running the notebook.

---

## Machine Learning Pipeline

The experimental pipeline follows these steps:

1. Load SAML-D dataset
2. Explore dataset information and missing values
3. Analyze class distribution
4. Perform subsampling due to large dataset size
5. Conduct feature engineering (temporal, sender behavior, currency mismatch)
6. Remove data leakage columns
7. Encode categorical variables using label encoding
8. Scale numerical features using StandardScaler
9. Split dataset into training and testing sets (80:20 stratified)
10. Apply RandomOverSampler on the training set only (sampling strategy = 0.1)
11. Train Random Forest, XGBoost, and LightGBM
12. Evaluate classification performance
13. Interpret the best model using SHAP
14. Benchmark inference performance for Edge AI deployment

---

## Model Evaluation

### Classification Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Random Forest | 0.9087 | 0.2960 | 0.6825 | 0.4129 | 0.9066 | 0.5718 |
| XGBoost | 0.8048 | 0.1783 | 0.8724 | 0.2961 | 0.9263 | 0.6374 |
| **LightGBM** | **0.8782** | **0.2504** | **0.7970** | **0.3811** | **0.9261** | **0.6382** |

> LightGBM is selected as the main XE-AML edge model because it achieves the highest PR-AUC and provides a strong balance between detection quality and computational efficiency. PR-AUC is emphasized over Accuracy because the dataset is highly imbalanced.

### Class Distribution
![Class Distribution](pencucian%20uang/pencucian%20uang/class_distribution.png)

### Amount Distribution
![Amount Distribution](pencucian%20uang/pencucian%20uang/amount_distribution.png)

### Model Comparison
![Model Comparison](pencucian%20uang/pencucian%20uang/model_comparison.png)

### Confusion Matrix
![Confusion Matrix](pencucian%20uang/pencucian%20uang/confusion_matrices.png)

### Precision-Recall Curve
![Precision Recall Curve](pencucian%20uang/pencucian%20uang/precision_recall_curve.png)

### ROC Curve
![ROC Curve](pencucian%20uang/pencucian%20uang/roc_curve.png)

---

## SHAP Explainability

SHAP is used to explain how the trained LightGBM model makes predictions. AML decisions must not only be accurate, but also transparent, auditable, and understandable for compliance officers and regulators. SHAP provides both global and local explanations for suspicious transaction detection.

Key features identified by SHAP include: transaction amount, sender transaction count, sender amount deviation, currency mismatch, and payment type.

### SHAP Summary Plot
![SHAP Summary Plot](pencucian%20uang/pencucian%20uang/shap_summary_plot.png)

### SHAP Feature Importance
![SHAP Bar Plot](pencucian%20uang/pencucian%20uang/shap_bar_plot.png)

### SHAP Dependence Plot
![SHAP Dependence Plot](pencucian%20uang/pencucian%20uang/shap_dependence_plot.png)

### SHAP Waterfall — Suspicious Transaction
![SHAP Waterfall Suspicious](pencucian%20uang/pencucian%20uang/shap_waterfall_suspicious.png)

### SHAP Waterfall — Normal Transaction
![SHAP Waterfall Normal](pencucian%20uang/pencucian%20uang/shap_waterfall_normal.png)

---

## Inference Benchmark and Computational Performance

Computational performance is evaluated to determine whether the LightGBM model can operate efficiently on edge infrastructure. Low latency and small model size are particularly important because the proposed system aims to detect suspicious transactions before funds move further through the financial system.

The benchmark measures four core metrics defined in Section 3.5 of the paper:

- **Inference latency** — time per transaction (ms)
- **Memory consumption** — RSS delta when loading the model (MB)
- **Model size** — serialized file size on disk (MB)
- **Transaction throughput** — predictions per second (TPS)

Additionally, CPU usage is measured using a background polling monitor normalized per logical core.

### Benchmark Environment

| Parameter | Value |
|---|---|
| Platform | Windows |
| CPU logical cores | 8 |
| CPU physical cores | 6 |
| RAM total | 7.71 GB |
| Python version | 3.12.9 |
| Warm-up inferences | 200 |
| Measurement samples | 1,000 single-row inferences |

### Benchmark Results — LightGBM

| Metric | Value | Interpretation |
|---|---:|---|
| **Mean Latency** | **1.2647 ms** | Average inference time per transaction |
| **Median Latency** | **1.1640 ms** | Typical inference time |
| **P95 Latency** | **1.7895 ms** | 95% of transactions processed within this limit (SLA target) |
| **P99 Latency** | **2.2710 ms** | Upper-bound near-real-time inference latency (SLA ceiling) |
| **Min / Max Latency** | **0.9042 / 2.8022 ms** | Observed latency range |
| **Model Size** | **1.015 MB** | Lightweight — suitable for edge deployment |
| **Memory Footprint** | **2.85 MB** | Additional RSS memory required when loading model + scaler |
| **Inference Memory Delta** | **4.70 MB** | Additional memory consumed during active inference loop |
| **Batch Throughput** | **169,072 TPS** | High-volume batch inference capability (5,000 samples) |
| **Stress Test Throughput** | **247,767 TPS** | Transaction processing capacity under sustained load (≥5 s loop) |
| **Avg CPU per-core** | **72.90%** | Normalized per logical core — valid measurement over 5-second loop |
| **CPU samples valid** | **48 / 48** | All monitor samples captured successfully |

> **Edge AI Suitability:** All six criteria passed ✅  
> Model Size < 10 MB · Memory Footprint < 50 MB · Mean Latency < 5 ms · P99 Latency < 10 ms · Throughput > 1,000 TPS · Avg CPU per-core < 80%

### Notes on Measurement Methodology

**Memory Footprint** is measured as the RSS (Resident Set Size) delta before and after loading the model — not the total process memory. This isolates the actual memory cost attributable to the model itself.

**CPU Usage** is measured using `psutil` in a background polling thread (interval: 100 ms) during a sustained inference loop of at least 5 seconds. The raw value from `psutil.cpu_percent()` on Windows reports cumulative CPU across all cores (e.g., 8 cores × ~79% ≈ 583%). This value is normalized by dividing by the number of logical cores to obtain per-core utilization (583% ÷ 8 = **72.90% per-core**). A warm-up phase of 200 inferences is performed before measurement to eliminate JIT cold-start overhead.

**P95 and P99 latency** are more important than mean latency for production SLA assessment because they capture worst-case performance under realistic conditions.

### Benchmark Files

```text
pencucian uang/pencucian uang/
├── inference_benchmark.py          ← standalone benchmark script
├── inference_metrics_updated.json  ← latest benchmark results (JSON)
└── aml_detection_FINAL (3).ipynb  ← Section 10: notebook benchmark cells
```

---

## Repository Contents

```text
README.md
.gitignore

pencucian uang/pencucian uang/
├── aml_detection_FINAL (3).ipynb
├── inference_benchmark.py
├── inference_metrics_updated.json
├── inference_metrics.json.json
├── aml_best_model.pkl
├── aml_scaler.pkl
├── class_distribution.png
├── amount_distribution.png
├── model_comparison.png
├── confusion_matrices.png
├── precision_recall_curve.png
├── roc_curve.png
├── feature_importance.png
├── shap_summary_plot.png
├── shap_bar_plot.png
├── shap_dependence_plot.png
├── shap_waterfall_suspicious.png
└── shap_waterfall_normal.png
```

---

## Paper Context

This implementation supports the innovation paper:

> **"Explainable Edge AI for Cost-Efficient and Trustworthy Anti-Money Laundering Systems"**  
> Bella Adisty, Nela Eka Silvia Lumbanraja, Albert Jofrandi Hutapea  
> Program Studi Informatika, Institut Teknologi Sains Bandung  
> Innovation Paper Competition 2026 — Risk and Governance Summit

The proposed XE-AML framework combines:

- **Edge AI** for low-latency, near-real-time transaction screening before funds move further
- **LightGBM** for lightweight machine learning inference with small model footprint
- **SHAP** for explainable, auditable AML decision-making
- **Inference benchmarking** to evaluate deployment feasibility on edge infrastructure
- **Computational efficiency analysis** using latency, throughput, model size, memory, and CPU metrics

The computational evaluation supports the claim in Section 3.5 of the paper: LightGBM achieves a mean inference latency of **1.26 ms**, a model size of **1.015 MB**, a memory footprint of **2.85 MB**, and a transaction throughput of **169,072 TPS** — all within the thresholds required for efficient edge deployment.

---

## Notes

- The dataset file is excluded from the repository due to GitHub's file size limitation.
- The trained model (`aml_best_model.pkl`) and scaler (`aml_scaler.pkl`) are included because their combined size is only 1.017 MB.
- The notebook may not always render correctly on GitHub if the output is large. Result figures are provided separately as PNG files.
- CPU usage is reported as per-core normalized value. Raw psutil output on multi-core Windows systems may show values above 100% (cumulative across all cores).
