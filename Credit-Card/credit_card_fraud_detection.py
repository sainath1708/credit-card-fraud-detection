# Credit Card Fraud Detection Project

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_curve, auc
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
print("Loading dataset...")
df = pd.read_csv("C:\\Users\\Sainath\\Downloads\\archive (1)\\creditcard.csv")

# Exploratory Data Analysis
print("\nDataset Information:")
print(df.info())
print("\nFirst few rows:")
print(df.head())
print("\nBasic statistics:")
print(df.describe())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Check class distribution
print("\nClass distribution:")
print(df['Class'].value_counts())
fraud_percent = df['Class'].value_counts()[1] / len(df) * 100
print(f"Percentage of fraudulent transactions: {fraud_percent:.4f}%")

# Data Visualization
plt.figure(figsize=(10, 8))
sns.countplot(x='Class', data=df)
plt.title('Class Distribution')
plt.xlabel('Class (0: Normal, 1: Fraud)')
plt.ylabel('Count')
plt.savefig('class_distribution.png')
plt.close()

# Distribution of transaction amounts
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(df[df['Class'] == 0]['Amount'], kde=True)
plt.title('Distribution of Transaction Amounts (Normal)')
plt.xlabel('Amount')
plt.ylabel('Frequency')
plt.xlim([0, 500])

plt.subplot(1, 2, 2)
sns.histplot(df[df['Class'] == 1]['Amount'], kde=True, color='red')
plt.title('Distribution of Transaction Amounts (Fraud)')
plt.xlabel('Amount')
plt.ylabel('Frequency')
plt.savefig('amount_distribution.png')
plt.close()

# Correlation matrix
plt.figure(figsize=(16, 14))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False)
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.png')
plt.close()

# Feature scaling
scaler = StandardScaler()
df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))

# Drop original Time and Amount columns
df = df.drop(['Time', 'Amount'], axis=1)

# Prepare the data for modeling
X = df.drop('Class', axis=1)
y = df['Class']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Handle imbalanced data using SMOTE
print("\nApplying SMOTE to handle class imbalance...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"After SMOTE - Training set shape: {X_train_smote.shape}")
print(f"Class distribution after SMOTE: {pd.Series(y_train_smote).value_counts()}")

# Train Logistic Regression model
print("\nTraining Logistic Regression model...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_smote, y_train_smote)

# Train Random Forest model
print("Training Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_smote, y_train_smote)

# Evaluate models
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    print(f"\n{model_name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Classification Report:\n{class_report}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
    plt.close()
    
    # Precision-Recall curve
    y_scores = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend(loc='best')
    plt.savefig(f'pr_curve_{model_name.lower().replace(" ", "_")}.png')
    plt.close()
    
    return accuracy, pr_auc

# Evaluate both models
lr_accuracy, lr_pr_auc = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
rf_accuracy, rf_pr_auc = evaluate_model(rf_model, X_test, y_test, "Random Forest")

# Compare models
print("\nModel Comparison:")
models = ['Logistic Regression', 'Random Forest']
accuracies = [lr_accuracy, rf_accuracy]
pr_aucs = [lr_pr_auc, rf_pr_auc]

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.bar(models, accuracies, color=['blue', 'green'])
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim([0, 1])

plt.subplot(1, 2, 2)
plt.bar(models, pr_aucs, color=['blue', 'green'])
plt.title('Model PR-AUC Comparison')
plt.ylabel('PR-AUC')
plt.ylim([0, 1])

plt.tight_layout()
plt.savefig('model_comparison.png')
plt.close()

# Feature importance for Random Forest
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
plt.title('Top 10 Feature Importance - Random Forest')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

print("\nCredit Card Fraud Detection project completed successfully!")
print("Check the generated visualizations for insights.") 