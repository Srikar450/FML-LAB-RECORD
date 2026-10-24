
import pandas as pd
from sklearn.model_selection import train_test_split

# Step 1: Load the dataset
data = pd.read_csv("data.csv", sep="\t", encoding="utf-8-sig")

# Step 2: Remove extra spaces from column names
data.columns = data.columns.str.strip()

# Step 3: Display column names
print("Column Names:")
print(data.columns.tolist())

# Step 4: Define required columns
required_columns = [
    "Hours_Studied",
    "Attendance",
    "Assignments",
    "Result"
]

# Step 5: Check for missing columns
missing_columns = [
    col for col in required_columns
    if col not in data.columns
]

if missing_columns:
    print("\nMissing columns:", missing_columns)
    print("Please check your CSV file.")

else:
    # Step 6: Select input features (X) and target (y)
    X = data[["Hours_Studied", "Attendance", "Assignments"]].copy()
    y = data["Result"].astype(str).str.strip()

    # Step 7: Convert input features to numeric values
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Step 8: Handle missing values
    dataset = pd.concat([X, y.rename("Result")], axis=1)
    dataset = dataset.dropna()

    X = dataset[["Hours_Studied", "Attendance", "Assignments"]]
    y = dataset["Result"]

    # Step 9: Check dataset size
    if len(dataset) < 2:
        print("\nNot enough valid rows to split the dataset.")

    else:
        # Step 10: Split into 80% training and 20% testing
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        # Step 11: Display training and testing data
        print("\nTraining Data:")
        print(X_train)

        print("\nTesting Data:")
        print(X_test)

        print("\nTraining Target Values:")
        print(y_train)

        print("\nTesting Target Values:")
        print(y_test)

        print("\nTraining Samples:", len(X_train))
        print("Testing Samples:", len(X_test))

        print("\nDataset split completed successfully!")
