import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Step 1: Extract - Load data from raw CSV
df = pd.read_csv('raw_data.csv')
print("Raw Data:\n", df)

# Step 2: Preprocess - Handle missing values

# Impute missing numeric values (like age and salary) with mean
imputer = SimpleImputer(strategy='mean')
df[['age', 'salary']] = imputer.fit_transform(df[['age', 'salary']])

# Step 3: Transform - Standardize numerical columns

scaler = StandardScaler()
df[['age', 'salary']] = scaler.fit_transform(df[['age', 'salary']])

print("\nTransformed Data:\n", df)

# Step 4: Load - Save the cleaned data to a new CSV
df.to_csv('cleaned_data.csv', index=False)
print("\nCleaned data has been saved to 'cleaned_data.csv'")
