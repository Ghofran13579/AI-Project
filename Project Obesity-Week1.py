# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Task 1: Import the dataset and inspect its structure
# Load the dataset
df = pd.read_csv(r"C:\Users\massa\OneDrive\Desktop\All Folders\HubbleMind\ObesityDataSet_raw_and_data_sinthetic.csv")

# Inspect the first few rows
print("First few rows of the dataset:")
print(df.head())

# Check the data types of columns
print("\nData types of the columns:")
print(df.dtypes)

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Task 2: Data Type Conversion and Encoding
# Initialize LabelEncoder for binary variables
label_encoder = LabelEncoder()

# Label encode binary variables like 'Gender' and 'SMOKE'
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['SMOKE'] = label_encoder.fit_transform(df['SMOKE'])

# One-hot encode multi-class categorical variables like 'MTRANS' and 'NObeyesdad'
df = pd.get_dummies(df, columns=['MTRANS', 'NObeyesdad'], drop_first=True)

# Inspect the data after encoding
print("\nData after encoding:")
print(df.head())

# Task 3: Outlier Detection and Handling
# Boxplots for detecting outliers in continuous variables 'Weight' and 'Height'
plt.figure(figsize=(12, 6))

# Boxplot for Weight
plt.subplot(1, 2, 1)
sns.boxplot(x=df['Weight'])
plt.title('Boxplot for Weight')

# Boxplot for Height
plt.subplot(1, 2, 2)
sns.boxplot(x=df['Height'])
plt.title('Boxplot for Height')

plt.tight_layout()
plt.show()

# Handling outliers by capping at the 1st and 99th percentiles for 'Weight' and 'Height'
lower_limit_weight = df['Weight'].quantile(0.01)
upper_limit_weight = df['Weight'].quantile(0.99)
df['Weight'] = df['Weight'].clip(lower=lower_limit_weight, upper=upper_limit_weight)

lower_limit_height = df['Height'].quantile(0.01)
upper_limit_height = df['Height'].quantile(0.99)
df['Height'] = df['Height'].clip(lower=lower_limit_height, upper=upper_limit_height)

# Check the data after handling outliers
print("\nData after handling outliers:")
print(df.head())

# Task 4: Normalization/Standardization
# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Normalize continuous variables 'Age', 'Weight', 'Height'
df[['Age', 'Weight', 'Height']] = scaler.fit_transform(df[['Age', 'Weight', 'Height']])

# Check the data after normalization
print("\nData after normalization:")
print(df.head())

# Save the cleaned data to a new CSV file
df.to_csv(r"C:\Users\massa\OneDrive\Desktop\All Folders\HubbleMind\ObesityDataSet_cleaned.csv", index=False)
