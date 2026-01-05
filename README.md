# Promotion Response Prediction

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
- Evaluation metric: ROC AUC
- Validation: Stratified 80/20 train–validation split
- Regularization: Early stopping, subsampling, tree constraints

Neural networks were not used due to overfitting risk on tabular data.

---

## Results
- Training AUC: 0.875  
- Validation AUC: 0.713  

Key drivers of promotion response:
1. Customer engagement intensity
2. Recency of activity
3. Brand affinity
4. Store-level contextual information

---

## Project Structure

project-promotion-response-prediction/
├── data/          # Contains raw data (ignored by GitHub)
├── notebook/      # Jupyter notebook with model code
├── report/        # PDF report summarizing approach and results
├── output/        # Generated predictions CSV (ignored by GitHub)
├── README.md      # This file
└── .gitignore     # Files/folders to exclude from GitHub


---

## Reproducibility
- Notebook fully executable on **Google Colab**  
- Runtime < 10 minutes for model training and predictions  
- Predictions are generated directly from the notebook without using external data  

---

## Future Work
- Implement rolling temporal features and trend analysis  
- Explore customer segmentation or cohort-based modeling for personalized targeting  
- Apply ensemble methods and automated feature selection to improve predictive accuracy  
- Use SHAP or similar techniques for enhanced feature interpretability  

---

## AI Usage Disclosure
Generative AI tools (ChatGPT, Claude) were used for **code drafting, debugging suggestions, and documentation clarification**. All outputs were independently reviewed, validated, and adapted for the project.

---

