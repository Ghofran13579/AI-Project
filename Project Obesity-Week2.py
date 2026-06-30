# Import necessary libraries for EDA
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Task 1: Import the dataset
# Replace 'path_to_your_data.csv' with the actual file path
df = pd.read_csv(r"C:\Users\massa\OneDrive\Desktop\All Folders\HubbleMind\ObesityDataSet_cleaned.csv")

# Print the first few rows to inspect the data
print("First few rows of the dataset:")
print(df.head())

# Task 1: Summary Statistics
# Generate summary statistics for continuous variables
summary_stats = df.describe()
print("\nSummary Statistics for continuous variables:")
print(summary_stats)

# Task 2: Distribution Analysis
# Plot histograms and KDE (Kernel Density Estimate) plots for Age, Weight, and Height
plt.figure(figsize=(15, 10))

# Histogram and KDE plot for Age
plt.subplot(2, 3, 1)
sns.histplot(df['Age'], kde=True, color='blue')
plt.title('Age Distribution')

# Histogram and KDE plot for Weight
plt.subplot(2, 3, 2)
sns.histplot(df['Weight'], kde=True, color='green')
plt.title('Weight Distribution')

# Histogram and KDE plot for Height
plt.subplot(2, 3, 3)
sns.histplot(df['Height'], kde=True, color='orange')
plt.title('Height Distribution')

# Adjust layout
plt.tight_layout()
plt.show()

# Task 3: Relationship Exploration
# Use boxplots to explore relationships between features (e.g., Weight and FAF) and obesity levels
# Assuming 'FAF' is a feature (you should replace it with the actual feature name from your dataset)
plt.figure(figsize=(15, 7))

# Boxplot to explore the relationship between Weight and obesity levels (assuming obesity levels are in 'NObeyesdad_Obesity_Type_I')
plt.figure(figsize=(15, 7))

# Boxplot for Weight vs Obesity Type I
plt.subplot(1, 2, 1)
sns.boxplot(x=df['NObeyesdad_Obesity_Type_I'], y=df['Weight'])
plt.title('Weight vs Obesity Type I')

# Boxplot for FAF vs Obesity Type I (replace 'FAF' with another relevant feature if necessary)
plt.subplot(1, 2, 2)
sns.boxplot(x=df['NObeyesdad_Obesity_Type_I'], y=df['FAF'])  # Replace 'FAF' with another feature if needed
plt.title('FAF vs Obesity Type I')

plt.tight_layout()
plt.show()

# Task 1: Advanced Visualizations
# Pairplots for feature analysis
sns.pairplot(df, hue='NObeyesdad_Obesity_Type_I', palette='coolwarm', diag_kind='kde')
plt.suptitle('Pairplot Analysis', y=1.02)
plt.show()

# Heatmap of the confusion matrix (to be generated after model predictions)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Dummy setup to visualize feature importance (adjust 'target_column' accordingly)
X = df.drop('NObeyesdad_Obesity_Type_I', axis=1)  # Drop the target column
y = df['NObeyesdad_Obesity_Type_I']
rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)

# Feature importance plot
importances = rf.feature_importances_
features = X.columns
plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=features, palette='viridis')
plt.title('Feature Importance - Random Forest')
plt.show()