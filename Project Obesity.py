# Import necessary libraries for EDA and ML
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

# Task 1: Import the dataset
df = pd.read_csv(r"C:\Users\massa\OneDrive\Desktop\All Folders\HubbleMind\ObesityDataSet_cleaned.csv")

# Print the first few rows to inspect the data
print("First few rows of the dataset:")
print(df.head())

# Task 1: Summary Statistics
summary_stats = df.describe()
print("\nSummary Statistics for continuous variables:")
print(summary_stats)

# Additional Summary Statistics
continuous_columns = ['Age', 'Weight', 'Height']  # Replace with relevant continuous variables
for col in continuous_columns:
    print(f"\nAdditional Statistics for {col}:")
    print(f"Median: {df[col].median()}")
    print(f"Mode: {df[col].mode()[0]}")

# Task 2: Distribution Analysis
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
sns.histplot(df['Age'], kde=True, color='blue')
plt.title('Age Distribution')

plt.subplot(2, 3, 2)
sns.histplot(df['Weight'], kde=True, color='green')
plt.title('Weight Distribution')

plt.subplot(2, 3, 3)
sns.histplot(df['Height'], kde=True, color='orange')
plt.title('Height Distribution')

plt.tight_layout()
plt.show()

# Task 3: Relationship Exploration
plt.figure(figsize=(15, 7))

plt.subplot(1, 2, 1)
sns.boxplot(x=df['NObeyesdad_Obesity_Type_I'], y=df['Weight'])
plt.title('Weight vs Obesity Type I')

plt.subplot(1, 2, 2)
sns.boxplot(x=df['NObeyesdad_Obesity_Type_I'], y=df['FAF'])
plt.title('FAF vs Obesity Type I')

plt.tight_layout()
plt.show()

# Task 4: Correlation Analysis
correlation_matrix = df[['Age', 'Weight', 'Height']].corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", cbar=True)
plt.title('Correlation Heatmap')
plt.show()

# Task 1: Advanced Visualizations
# Pairplot for key variables
sns.pairplot(df, vars=['Age', 'Weight', 'Height'], hue='NObeyesdad_Obesity_Type_I', palette='husl')
plt.suptitle('Pairplot of Key Variables', y=1.02)
plt.show()

# Task 2: Feature Engineering and Scaling
# Encoding categorical columns
label_encoders = {}
categorical_columns = ['Gender', 'NObeyesdad_Obesity_Type_I']  # Add all necessary categorical columns
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Scaling continuous features
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

# Ensure all categorical columns are encoded
label_encoders = {}
for col in categorical_columns:
    if col != 'NObeyesdad_Obesity_Type_I':  # Skip the target column
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Scale continuous features
continuous_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
continuous_columns.remove('NObeyesdad_Obesity_Type_I')  # Remove target column if included
scaler = StandardScaler()
df[continuous_columns] = scaler.fit_transform(df[continuous_columns])

# Task 3: Train-Test Split
X = df.drop(columns=['NObeyesdad_Obesity_Type_I'])  # Features
y = df['NObeyesdad_Obesity_Type_I']  # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Task 4: Machine Learning Model Implementation
# Logistic Regression
log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train, y_train)

# Random Forest
rf_clf = RandomForestClassifier(random_state=42, n_estimators=100)
rf_clf.fit(X_train, y_train)

# Task 5: Model Evaluation
# Logistic Regression Evaluation
y_pred_log_reg = log_reg.predict(X_test)
print("\nLogistic Regression Evaluation:")
print(classification_report(y_test, y_pred_log_reg))
print("Accuracy:", accuracy_score(y_test, y_pred_log_reg))

# Random Forest Evaluation
y_pred_rf = rf_clf.predict(X_test)
print("\nRandom Forest Evaluation:")
print(classification_report(y_test, y_pred_rf))
print("Accuracy:", accuracy_score(y_test, y_pred_rf))

# Confusion Matrix for Random Forest
conf_matrix = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Feature Importance Plot for Random Forest
importances = rf_clf.feature_importances_
feature_names = X.columns
plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=feature_names, palette='viridis')
plt.title('Feature Importances - Random Forest')
plt.show()
