
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Step 1: Load the tab-separated dataset
data = pd.read_csv("data.csv", sep="\t", encoding="utf-8-sig")

# Step 2: Clean column names
data.columns = data.columns.str.strip()

print("Original Dataset:")
print(data.head())

# Step 3: Check missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Step 4: Select numeric columns
columns = [
    "Hours_Studied",
    "Attendance",
    "Assignments",
    "Marks"
]

# Step 5: Convert columns to numeric
for col in columns:
    data[col] = pd.to_numeric(data[col], errors="coerce")

X = data[columns].copy()

# Step 6: Fill missing values with column means
X = X.fillna(X.mean())

# Step 7: Check for columns with no valid values
if X.isnull().any().any():
    print("\nError: Some columns contain no valid numeric values.")

else:
    # Step 8: Normalize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 9: Display normalized dataset
    X_scaled = pd.DataFrame(
        X_scaled,
        columns=columns
    )

    print("\nNormalized Dataset:")
    print(X_scaled.head())

    print("\nData preprocessing completed successfully.")
