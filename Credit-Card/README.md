# Credit Card Fraud Detection

This project implements a machine learning solution for detecting fraudulent credit card transactions using various classification algorithms.

## Project Overview

The project uses a dataset of credit card transactions to build and evaluate machine learning models for fraud detection. The dataset contains transactions made by credit cards in September 2013 by European cardholders. The dataset is highly imbalanced, with only 0.172% of transactions being fraudulent.

## Features

- Exploratory Data Analysis (EDA)
- Data Visualization
- Feature Engineering
- Handling Class Imbalance using SMOTE
- Model Training and Evaluation
- Performance Metrics and Visualizations

## Requirements

- Python 3.8+
- Required packages are listed in `requirements.txt`

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Dataset

The dataset used in this project is available on Kaggle:
[Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)

Please download the dataset and place it in the project directory.

## Project Structure

- `credit_card_fraud_detection.py`: Main script containing the implementation
- `requirements.txt`: List of required Python packages
- `README.md`: Project documentation
- Generated visualizations:
  - `class_distribution.png`: Distribution of normal vs. fraudulent transactions
  - `amount_distribution.png`: Distribution of transaction amounts
  - `correlation_matrix.png`: Correlation between features
  - `confusion_matrix_*.png`: Confusion matrices for different models
  - `pr_curve_*.png`: Precision-Recall curves for different models
  - `model_comparison.png`: Comparison of model performances
  - `feature_importance.png`: Feature importance from Random Forest

## Implementation Details

The project includes:
1. Data preprocessing and feature scaling
2. Handling class imbalance using SMOTE
3. Training of two models:
   - Logistic Regression
   - Random Forest Classifier
4. Model evaluation using various metrics
5. Generation of visualizations for analysis

## Results

The project evaluates models based on:
- Accuracy
- Confusion Matrix
- Classification Report
- Precision-Recall Curve
- Feature Importance

## Usage

1. Download the dataset from Kaggle
2. Place the dataset in the project directory
3. Run the main script:
```bash
python credit_card_fraud_detection.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset provided by [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Special thanks to the machine learning community for their contributions 