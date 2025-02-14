# 🏡 Predictive Analytics on Irish House Prices

## 📌 Project Overview
This project implements predictive analytics techniques to analyze housing prices in Dublin, Ireland. The objective is to identify key factors that influence a homebuyer’s decision and develop machine learning models to predict the likelihood of purchasing a property based on location, price per square foot, insulation, and other features.

## 📊 Dataset
The dataset consists of 13,320 rows and 12 columns, capturing property details such as:
- **Location** (Four Dublin local authorities)
- **Property Size** (Number of bedrooms)
- **Price per Square Foot** ($)
- **Baths, Balconies, and BER Ratings**
- **Renovation Requirements** (Yes/No/Maybe)

## 🔍 Key Steps in the Project
### 1️⃣ **Exploratory Data Analysis (EDA)**
- Distribution analysis using histograms and box plots.
- Correlation matrix visualization.
- Missing value heatmaps.

### 2️⃣ **Data Cleaning & Feature Engineering**
- Imputation of missing values (median/mode strategy).
- Conversion of categorical variables (e.g., BER rating to numerical codes).
- Creation of derived features (e.g., price range classification, location-based price averages).

### 3️⃣ **Outlier Detection & Handling**
- Interquartile Range (IQR) method to cap extreme values.
- Z-score computation to analyze price deviations.
- Scatterplots and pairplots for anomaly visualization.

### 4️⃣ **Predictive Modeling**
- **Decision Tree Classifier** (Baseline)
- **Random Forest** (Hyperparameter tuning with GridSearchCV)
- **Voting Classifier** (Ensemble learning combining RF, Gradient Boosting, and Logistic Regression)
- **Bagging Classifier** (Enhancing model stability)
- **SMOTE** (Synthetic Minority Over-sampling Technique) to balance class distributions.

### 5️⃣ **Model Evaluation**
- Accuracy, Precision, Recall, and ROC-AUC scores.
- Cross-validation to validate generalization.
- ROC Curve Comparison across all models.

## ⚡ Results
- **Voting Classifier** outperformed others with **75% accuracy** and an **ROC-AUC score of 0.61**.
- Random Forest with hyperparameter tuning provided a robust alternative.
- SMOTE significantly improved Decision Tree performance.

## 🛠️ Tech Stack
- **Python 3.8+**
- `pandas`, `numpy` - Data Manipulation
- `matplotlib`, `seaborn`, `missingno` - Data Visualization
- `scikit-learn`, `imblearn` - Machine Learning & Model Evaluation

## 🚀 Getting Started
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/irish-house-price-prediction.git
