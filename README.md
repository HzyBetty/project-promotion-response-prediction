# Promotion Response Prediction

## Quick Summary

| | |
|---|---|
| **Business goal** | Rank customers by response likelihood to focus promotion budget on the most likely engagers |
| **Model** | LightGBM gradient boosting |
| **Validation AUC** | 0.713 (vs 0.50 random baseline) |
| **Top-20% targeting lift** | ~2× more responders than random outreach |
| **Recommended threshold** | ~0.21 (maximises F1 given 80/20 class imbalance) |
| **Features engineered** | 71 (RFM behaviour, brand affinity, store context, price sensitivity) |

## Overview
This project builds a machine learning model to predict whether a customer will respond to a retail promotion. Using historical transaction data and promotion metadata, the model estimates the probability of customer engagement with future promotional offers. Accurate prediction enables improved campaign targeting, reduced marketing waste, and higher ROI.

The task is formulated as a binary classification problem and evaluated using ROC AUC on a blind test set.

---

## Dataset
The dataset consists of anonymized retail data (excluded from this repo) and includes:

- **transactions.parquet**: Historical customer transactions
- **promos.parquet**: Metadata describing each promotion
- **train_history.parquet**: Promotions offered in March 2013 with observed responses
- **test_history.parquet**: Promotions offered in April 2013 without response labels
- **data_dictionary.xlsx**: Detailed field-level documentation

All features are engineered using **strictly pre-promotion data** to prevent temporal leakage.

---

## Methodology

### Feature Engineering
- RFM behavior: recency, frequency, monetary value at customer, brand, category, and manufacturer levels
- Brand & category affinity: loyalty scores, spend shares, purchase diversity
- Store-level context: normalized behavioral ratios relative to store averages
- Temporal signals: day-of-week, month, weekend indicators
- Price sensitivity: spending variability, response to promotion value and quantity
- Competitive pressure: recent promotions in the same store and category

High-cardinality categorical variables were **smoothed target-encoded**, and log transforms, ratios, and interaction features were applied to improve robustness.

### Model
- Algorithm: LightGBM (gradient boosting decision trees)
- Benchmarked against: Logistic Regression and Random Forest baselines
- Evaluation metric: ROC AUC
- Validation: Stratified 80/20 split confirmed with 5-fold cross-validation (AUC 0.71 ± 0.01)
- Regularization: Early stopping (patience 50), feature subsampling, L1/L2 penalties, tree depth limits

Neural networks were not used due to overfitting risk on tabular data with this sample size.

---

## Results

| Metric | Value |
|---|---|
| Training AUC | 0.875 |
| Validation AUC | 0.713 |
| 5-fold cross-validated AUC | 0.71 ± 0.01 |
| Top-20% capture rate | ~2× lift over random |
| Recommended decision threshold | ~0.21 (maximises F1) |

Key drivers of promotion response:
1. **Day-of-week purchase patterns** — counts of past transactions per weekday are the single most predictive feature class, reflecting how well a customer's shopping schedule aligns with the promotion
2. **Spending variability and magnitude** — coefficient of variation, mean, and standard deviation of transaction amounts capture how consistent and engaged a customer's spending is
3. **Brand and category diversity** — breadth of purchasing across brands and categories signals general shopping engagement
4. **Price sensitivity** — promotion value relative to the customer's typical transaction size (`promoval_vs_customer_avg`)
5. **Store-level and category-level engagement** — historical response rates and category spend provide contextual benchmarks

See `output/` for ROC curve, feature importance, SHAP explanations, lift/gain chart, calibration curve, and partial dependence plots.

---

## Reproducing This Analysis

All results in this repository are fully reproducible from the notebook. To replicate the reported numbers exactly:

1. **Use the same data files** — the four `.parquet` files from the course dataset. Results are tied to this specific dataset; substituting different data will produce different metrics.

2. **Run all cells in order from top to bottom.** The notebook is stateful: feature engineering, model training, evaluation, and visualisation cells must execute in sequence. Running individual cells out of order will cause `NameError` or produce stale results.

3. **Expected outputs after a full run:**
   - Training AUC: **0.8751**, Validation AUC: **0.7134**
   - 5-fold CV AUC: **0.7063 ± 0.0111**
   - F1-optimal threshold: **0.21** — flags 37.5% of customers, capturing 61.3% of responders at 33.1% precision
   - Top-20% lift: **2.05×** over random targeting
   - 10 chart images + `predict.csv` downloaded to your local machine

4. **Random seeds are fixed** (`random_state=42` throughout). Results should be numerically identical across runs on the same data, regardless of machine or Python environment, provided package versions meet the `requirements.txt` minimums.

5. **Runtime** on a free Colab CPU: approximately 5–10 minutes for feature engineering and training; SHAP computation adds another 1–2 minutes.

---

## How to Run

### Prerequisites

Obtain the four data files from your course materials and keep them together in one folder:

- `promos.parquet`
- `train_history.parquet`
- `test_history.parquet`
- `transactions.parquet`

These files are **not included in this repository**.

---

### Option A — Google Colab (recommended)

1. **Upload the data files to Google Drive.** Place all four `.parquet` files inside a single folder (e.g. `MyDrive/RSM8421_data`).

2. **Open the notebook in Colab.** Go to [colab.research.google.com](https://colab.research.google.com) → File → Open notebook → GitHub, paste this repo URL, and select `notebook/promotion_response_model.ipynb`. Alternatively, upload the `.ipynb` file directly.

3. **Set the data path.** In the cell directly below the **Load Data** heading, update the `path` variable to point to your Drive folder:
   ```python
   path = '/content/drive/MyDrive/RSM8421_data'  # ← change to match your folder name
   ```

4. **Run all cells in order.** Use **Runtime → Run all** (`Ctrl+F9`). When prompted, grant Drive access and complete the authorisation flow.

5. **Download outputs.** Prediction CSV and chart images are downloaded automatically at the end of each section via `files.download()`.

> **Runtime estimate:** Feature engineering on 30 k rows + LightGBM training + SHAP takes approximately 5–10 minutes on a free Colab CPU runtime.

---

### Option B — Local Jupyter

1. **Clone the repo** and place the four data files in the `data/` folder.

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Edit the data-loading cells.** Replace the Colab Drive block:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   path = '/content/drive/MyDrive/your_folder_name'
   ```
   with:
   ```python
   path = 'data'
   ```

4. **Skip the Colab download cells.** Two cells use `from google.colab import files` to trigger browser downloads — skip or comment them out when running locally.

5. **Launch the notebook** with `jupyter notebook` or open it in VS Code / JupyterLab.

---

All stochastic operations use `random_state=42`. The notebook was developed and validated on Google Colab (Python 3.10).

---

## If Deployed in Production

A score-based deployment would:
1. Re-run feature engineering on the latest transaction snapshot before each campaign
2. Apply the saved model to score all eligible customers
3. Rank customers by predicted probability and select those above the chosen threshold (~0.21 for F1-optimal targeting, adjustable based on campaign cost structure)
4. Monitor observed response rates of contacted customers to detect model drift over time

---

## Project Structure

```
project-promotion-response-prediction/
├── data/                          # Data files (not in repo — see How to Run)
│   ├── promos.parquet
│   ├── train_history.parquet
│   ├── test_history.parquet
│   └── transactions.parquet
├── notebook/
│   └── promotion_response_model.ipynb
├── output/                        # Generated charts and predictions (10 files)
│   ├── class_distribution.png
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   ├── lift_gain_chart.png
│   ├── calibration_curve.png
│   ├── partial_dependence.png
│   ├── feature_importance.png
│   ├── score_distribution.png
│   ├── shap_summary.png
│   └── shap_waterfall.png
├── report/
│   └── promotion_response_modeling_report.docx
├── requirements.txt
└── README.md
```
