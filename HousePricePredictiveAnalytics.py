import warnings
import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from scipy.stats import zscore
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
warnings.filterwarnings("ignore")


print('Data Load')
df_hpf_main=pd.read_csv('C:/Users/subra/Documents/Semester 1/IS 6052/Ireland_House_Price_Final.csv')

df_hpf = df_hpf_main.copy()
print(df_hpf)
df_hpf.head()
df_hpf.describe()
df_hpf.info()

print('Data Visualization')
numeric_columns = df_hpf.select_dtypes(include=['float64', 'int64']).columns

plt.figure(figsize=(10, 6))
sns.histplot(df_hpf['price-per-sqft-$'], kde=True, bins=30, color='blue')
plt.title('Distribution of Price per Square Foot')
plt.xlabel('Price per Square Foot ($)')
plt.ylabel('Frequency')
plt.show()

msno.heatmap(df_hpf, figsize=(10, 6), cmap='coolwarm', cbar=False)
plt.title('Missing Value Heatmap')
plt.show()

plt.figure(figsize=(12, 6))
df_hpf['location'].value_counts().plot(kind='bar', color='teal')
plt.title('Number of Properties by Location')
plt.xlabel('Location')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(12, 8))
sns.boxplot(x='location', y='price-per-sqft-$', data=df_hpf)
plt.xticks(rotation=45)
plt.title('Price per Square Foot by Location')
plt.xlabel('Location')
plt.ylabel('Price per Square Foot ($)')
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(df_hpf[numeric_columns].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()


print('Data Cleaning')
missingValues=df_hpf.isnull().sum()
print(missingValues)

df_hpf=df_hpf.dropna(subset=['location'])

print('After cleaning Location column')

df_hpf.rename(columns={'size':'size(No.of.Bedrooms)'},inplace=True)

df_hpf['size(No.of.Bedrooms)'] = df_hpf['size(No.of.Bedrooms)'].str.extract('(\d+)').astype(float)
print(df_hpf['size(No.of.Bedrooms)'])

df_hpf['size(No.of.Bedrooms)'].fillna(df_hpf['size(No.of.Bedrooms)'].median(), inplace=True)
print('After cleaning size(No.of.Bedrooms) column')

df_hpf['bath'].fillna(df_hpf['bath'].median(), inplace=True)
print('After cleaning bath column')

df_hpf['balcony'].fillna(0,inplace=True)
print('After cleaning balcony column')

df_hpf['price-per-sqft-$'].fillna(df_hpf['price-per-sqft-$'].median(), inplace=True)
print('After cleaning price-per-sqft-$ column')

df_hpf['total_sqft']=pd.to_numeric(df_hpf['total_sqft'].astype(str).str.replace('[^0-9.]', '',regex=True),errors='coerce')

df_hpf['total_sqft'].fillna(df_hpf['total_sqft'].median(), inplace=True)

df_hpf['property_scope'].fillna(df_hpf['property_scope'].mode()[0], inplace=True)

missingValues=df_hpf.isnull().sum()
print(missingValues)

numeric_columns = df_hpf.select_dtypes(include=['float64', 'int64']).columns

plt.figure(figsize=(10, 8))
sns.heatmap(df_hpf[numeric_columns].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

print("Data Cleaning Process over")

#feature_Engineering
df_hpf['NoOfRooms'] = df_hpf['size(No.of.Bedrooms)']+df_hpf['bath']

ber_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
df_hpf['BER_num'] = df_hpf['BER'].map(ber_mapping)

def price_range(price):
    if price < 300000:
        return 'Low'
    elif 300000 <= price < 600000:
        return 'Medium'
    else:
        return 'High'


df_hpf['price_range'] = df_hpf['price-per-sqft-$'] * df_hpf['total_sqft']
df_hpf['price_range'] = df_hpf['price_range'].apply(price_range)

price_range_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
df_hpf['priceRangeCode'] = df_hpf['price_range'].map(price_range_mapping)

df_hpf['renovationNeededCode'] = df_hpf['Renovation needed'].map({'Yes': 1, 'No': 0, 'Maybe': 0.5})

columns_to_drop = ['Renovation needed',  'BER',  'price_range']
df_hpf = df_hpf.drop(columns=columns_to_drop)

location_avg_price = df_hpf.groupby('location')['price-per-sqft-$'].mean()
df_hpf['location_avg_price'] = df_hpf['location'].map(location_avg_price)


def handle_rnge(value):
    if isinstance(value, str) and '-' in value:
        min_val, max_val = map(float, value.split('-'))
        return (min_val + max_val) / 2
    return value

df_hpf['total_sqft'] = df_hpf['total_sqft'].apply(handle_rnge)
df_hpf['total_sqft'] = pd.to_numeric(df_hpf['total_sqft'], errors='coerce')

plt.figure(figsize=(10, 6))
sns.boxplot(data=df_hpf[['price-per-sqft-$', 'total_sqft', 'NoOfRooms']], palette='Set3')
plt.title('Boxplots of Key Numerical Features')
plt.xlabel('Features')
plt.ylabel('Values')
plt.xticks(rotation=45)
plt.show()

sns.pairplot(df_hpf[['price-per-sqft-$', 'total_sqft', 'NoOfRooms']], diag_kind='kde', corner=True)
plt.suptitle('Pairplot for Outlier Detection', y=1.02)
plt.show()

df_hpf['z_score_price'] = zscore(df_hpf['price-per-sqft-$'])
plt.figure(figsize=(10, 6))
sns.histplot(df_hpf['z_score_price'], kde=True, bins=30, color='purple')
plt.axvline(3, color='red', linestyle='--', label='Z = 3')
plt.axvline(-3, color='red', linestyle='--', label='Z = -3')
plt.title('Z-Scores for Price-per-Square-Foot')
plt.xlabel('Z-Score')
plt.ylabel('Frequency')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='total_sqft', y='price-per-sqft-$', data=df_hpf, hue='priceRangeCode', palette='coolwarm', alpha=0.7)
plt.title('Scatterplot of Total Square Feet vs Price per Square Foot')
plt.xlabel('Total Square Feet')
plt.ylabel('Price per Square Foot ($)')
plt.legend(title='Price Range Code', loc='upper right')
plt.show()
# Outlier Handling
# Identify numerical columns
num_columns = df_hpf.select_dtypes(include=[np.number]).columns
# Visualize outliers using box plots
plt.figure(figsize=(15, 10))
df_hpf[num_columns].boxplot()
plt.title('Box Plots of Numerical Columns before Outliers Handling')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# Identify and exclude outliers using IQR
Q1 = df_hpf[['size(No.of.Bedrooms)', 'bath', 'balcony', 'total_sqft', 'price-per-sqft-$']].quantile(0.25)
Q3 = df_hpf[['size(No.of.Bedrooms)', 'bath', 'balcony', 'total_sqft', 'price-per-sqft-$']].quantile(0.75)
IQR = Q3 - Q1
# Define lower and upper bounds for non-outlier range for bath
lower_bound_bath = Q1['bath'] - 1.5 * IQR['bath']
upper_bound_bath = Q3['bath'] + 1.5 * IQR['bath']
df_hpf['bath'] = df_hpf['bath'].clip(lower=lower_bound_bath, upper=upper_bound_bath)
# Define lower and upper bounds for non-outlier range for bath
lower_bound_balcony = Q1['balcony'] - 1.5 * IQR['balcony']
upper_bound_balcony = Q3['balcony'] + 1.5 * IQR['balcony']
df_hpf['balcony'] = df_hpf['balcony'].clip(lower=lower_bound_balcony, upper=lower_bound_balcony)
# Define lower and upper bounds for non-outlier range for price-per-sqft-$
lower_bound_Price_per_squarefeet = Q1['price-per-sqft-$'] - 1.5 * IQR['price-per-sqft-$']
upper_bound_Price_per_squarefeet = Q3['price-per-sqft-$'] + 1.5 * IQR['price-per-sqft-$']
df_hpf['price-per-sqft-$'] = df_hpf['price-per-sqft-$'].clip(lower=lower_bound_Price_per_squarefeet, upper=upper_bound_Price_per_squarefeet)
# Define lower and upper bounds for non-outlier range for size(No.of.Bedrooms)
lower_bound_size = Q1['size(No.of.Bedrooms)'] - 1.5 * IQR['size(No.of.Bedrooms)']
upper_bound_size = Q3['size(No.of.Bedrooms)'] + 1.5 * IQR['size(No.of.Bedrooms)']
df_hpf['size(No.of.Bedrooms)'] = df_hpf['size(No.of.Bedrooms)'].clip(lower=lower_bound_size, upper=upper_bound_size)
# Define lower and upper bounds for non-outlier range for total_sqft
lower_bound_total_sqft = Q1['total_sqft'] - 1.5 * IQR['total_sqft']
upper_bound_total_sqft = Q3['total_sqft'] + 1.5 * IQR['total_sqft']
df_hpf['total_sqft'] = df_hpf['total_sqft'].clip(lower=lower_bound_total_sqft, upper=upper_bound_total_sqft)

print(f"Original dataset shape: {df_hpf.shape}")
print(f"Clean dataset shape: {df_hpf.shape}")

print("missing values in cleaned data",df_hpf.isnull().sum())




plt.figure(figsize=(15, 10))
df_hpf[num_columns].boxplot()
plt.title('Box Plots of Numerical Columns after Outlier Handling')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


print("missing values in cleaned data",df_hpf.isnull().sum())

#visualisation after outlier Handling
plt.figure(figsize=(10, 6))
sns.histplot(df_hpf['price-per-sqft-$'], kde=True, bins=30, color='blue')
plt.title('Distribution of Price per Square Foot')
plt.xlabel('Price per Square Foot ($)')
plt.ylabel('Frequency')
plt.show()

msno.heatmap(df_hpf, figsize=(10, 6), cmap='coolwarm', cbar=False)
plt.title('Missing Value Heatmap')
plt.show()

plt.figure(figsize=(12, 6))
df_hpf['location'].value_counts().plot(kind='bar', color='teal')
plt.title('Number of Properties by Location')
plt.xlabel('Location')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(12, 8))
sns.boxplot(x='location', y='price-per-sqft-$', data=df_hpf)
plt.xticks(rotation=45)
plt.title('Price per Square Foot by Location')
plt.xlabel('Location')
plt.ylabel('Price per Square Foot ($)')
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(df_hpf[numeric_columns].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

#creating data

le = LabelEncoder()
categorical_columns = ['property_scope', 'availability', 'location','buying or not buying']
for col in categorical_columns:
    df_hpf[col] = le.fit_transform(df_hpf[col])

df_hpf.dropna(inplace=True)

scaler = StandardScaler()
dfHpfScaled = df_hpf.copy()
dfHpfScaled[num_columns] = scaler.fit_transform(dfHpfScaled[num_columns])


dfHpfScaled.describe()
print(f"Scaled dataset shape: {dfHpfScaled.shape}")


X_buy = df_hpf.drop(['buying or not buying', 'ID'], axis=1)
y_buy = df_hpf['buying or not buying']


print('Rows for x_BUY' , X_buy.shape)
print('Rows for Y_buy', y_buy.shape)

X_buy = pd.get_dummies(X_buy, columns=['location', 'property_scope', 'availability'])

# Split the data
X_train_buy, X_test_buy, y_train_buy, y_test_buy = train_test_split(X_buy, y_buy, test_size=0.2, random_state=42)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_buy, y_train_buy)

print("Class distribution after SMOTE:")
print(y_train_smote.value_counts())

#predictive Modelling
def plot_roc_auc(y_true, y_pred_proba, model_name):
    """
    Plots ROC Curve and calculates AUC for a given model.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    return roc_auc

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
# Create and train the decision tree
dt_classifier = DecisionTreeClassifier(random_state=42)

dt_classifier.fit(X_train_buy, y_train_buy)

# Make predictions
y_pred = dt_classifier.predict(X_test_buy)

# Evaluate the model
print("Decision Tree Accuracy:", accuracy_score(y_test_buy, y_pred))
y_pred_proba_dt = dt_classifier.predict_proba(X_test_buy)[:, 1]

# Calculate and plot ROC-AUC
roc_auc_dt = plot_roc_auc(y_test_buy, y_pred_proba_dt, "Decision Tree")
print(f"Decision Tree AUC: {roc_auc_dt:.2f}")

#Decision tree with SMOTE
dt_classifier.fit(X_train_smote, y_train_smote)

# Make predictions
y_pred_smote = dt_classifier.predict(X_test_buy)

# Evaluate the model
print("Decision Tree Accuracy after smote:", accuracy_score(y_test_buy, y_pred_smote))

y_pred_proba_dt = dt_classifier.predict_proba(X_test_buy)[:, 1]

# Calculate and plot ROC-AUC
roc_auc_dt_smote= plot_roc_auc(y_test_buy, y_pred_proba_dt, "Decision Tree")
print(f"Decision Tree AUC: {roc_auc_dt_smote:.2f}")

#Creation of Random Forest Model
rf_classifier = RandomForestClassifier(n_estimators=200, max_depth=7, class_weight='balanced', random_state=42)

# Instantiate GridSearchCV
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Fit the grid search to the data
grid_search.fit(X_train_smote, y_train_smote)

# Print the best parameters and score
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

# Instantiate GridSearchCV
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

grid_search.fit(X_train_buy,y_train_buy)
# Use the best model for predictions
best_model = grid_search.best_estimator_
y_pred_buy_smote = best_model.predict(X_test_buy)

accuracy = accuracy_score(y_test_buy, y_pred_buy_smote)
print(f"Hyp Parameter tuning Random Forest Accuracy for buy after smote : {accuracy}")

# Fit the grid search to the data
grid_search.fit(X_train_buy, y_train_buy)

# Print the best parameters and score
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

# Use the best model for predictions
best_model = grid_search.best_estimator_
y_pred_buy = best_model.predict(X_test_buy)

y_pred_proba_rf = best_model.predict_proba(X_test_buy)[:, 1]

# Calculate and plot ROC-AUC
roc_auc_rf = plot_roc_auc(y_test_buy, y_pred_proba_rf, "Random Forest")
print(f" Hyp parameter tuning Random Forest AUC: {roc_auc_rf:.2f}")

accuracy = accuracy_score(y_test_buy, y_pred_buy)
print(f"Hyp Parameter tuning Random Forest Accuracy for buy: {accuracy}")

# Perform 5-fold cross-validation
cv_scores = cross_val_score(rf_classifier, X_train_buy, y_train_buy, cv=5)

print("Cross-validation scores:", cv_scores)
print("Mean CV score:", cv_scores.mean())
print("Standard deviation of CV scores:", cv_scores.std())

rf_classifier.fit(X_train_buy, y_train_buy)

# Make predictions
y_pred_buy = rf_classifier.predict(X_test_buy)

# Evaluate the model
accuracy = accuracy_score(y_test_buy, y_pred_buy)
print(f"Random Forest Accuracy for buy: {accuracy}")
print(classification_report(y_test_buy, y_pred_buy))


# Create base models
rf = RandomForestClassifier(n_estimators=200, max_depth=7, class_weight='balanced', random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
lr = LogisticRegression(random_state=42)

# Create voting classifier
voting_clf = VotingClassifier(
    estimators=[('rf', rf), ('gb', gb), ('lr', lr)],
    voting='soft'
)

# Fit and evaluate
voting_clf.fit(X_train_buy, y_train_buy)
voting_pred = voting_clf.predict(X_test_buy)
print("Voting Classifier Accuracy:", accuracy_score(y_test_buy, voting_pred))
print(classification_report(y_test_buy, voting_pred))

# Predict probabilities
y_pred_proba_voting = voting_clf.predict_proba(X_test_buy)[:, 1]

# Calculate and plot ROC-AUC
roc_auc_voting = plot_roc_auc(y_test_buy, y_pred_proba_voting, "Voting Classifier")
print(f"Voting Classifier AUC: {roc_auc_voting:.2f}")


bagging_clf = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=10,
    random_state=42
)

#bagging classifier
bagging_clf.fit(X_train_buy, y_train_buy)
bagging_pred = bagging_clf.predict(X_test_buy)
print("Bagging Classifier Accuracy:", accuracy_score(y_test_buy, bagging_pred))
print(classification_report(y_test_buy, bagging_pred))

# Predict probabilities
y_pred_proba_bagging = bagging_clf.predict_proba(X_test_buy)[:, 1]

# Calculate and plot ROC-AUC
roc_auc_bagging = plot_roc_auc(y_test_buy, y_pred_proba_bagging, "Bagging Classifier")
print(f"Bagging Classifier AUC: {roc_auc_bagging:.2f}")

plt.figure(figsize=(10, 6))

# Plot ROC curves for each model
roc_auc_dt = plot_roc_auc(y_test_buy, y_pred_proba_dt, "Decision Tree")
roc_auc_rf = plot_roc_auc(y_test_buy, y_pred_proba_rf, "Random Forest")
roc_auc_voting = plot_roc_auc(y_test_buy, y_pred_proba_voting, "Voting Classifier")
roc_auc_bagging = plot_roc_auc(y_test_buy, y_pred_proba_bagging, "Bagging Classifier")

# Finalize plot
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line
plt.title('ROC Curve Comparison')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()












