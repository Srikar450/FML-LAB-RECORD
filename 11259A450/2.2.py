
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Step 1: Load the tab-separated dataset
data = pd.read_csv("data.csv", sep="\t", encoding="utf-8-sig")

# Step 2: Remove extra spaces from column names
data.columns = data.columns.str.strip()

# Step 3: Check required columns
required_columns = [
    "Hours_Studied",
    "Attendance",
    "Assignments",
    "Result"
]

missing_columns = [
    col for col in required_columns
    if col not in data.columns
]

if missing_columns:
    print("Missing columns:", missing_columns)
    print("Available columns:", data.columns.tolist())

else:
    # Step 4: Select required columns
    data = data[required_columns].copy()

    # Step 5: Convert input features to numbers
    feature_columns = [
        "Hours_Studied",
        "Attendance",
        "Assignments"
    ]

    for col in feature_columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data["Result"] = data["Result"].astype(str).str.strip()
    data = data.dropna()
    data = data[data["Result"] != ""]

    X = data[feature_columns]
    y = data["Result"]

    # Step 6: Check dataset size and target classes
    if len(data) < 5 or y.nunique() < 2:
        print("Error: Need at least 5 valid rows and 2 target classes.")

    elif y.value_counts().min() < 5:
        print("Error: Each target class needs at least 5 rows for 5-fold cross-validation.")

    else:
        # Step 7: Split into training and testing data
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        print("Training Data Size:", len(X_train))
        print("Testing Data Size:", len(X_test))

        # Step 8: Create KNN model
        model = make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=3)
        )

        # Step 9: Perform 5-fold cross-validation
        scores = cross_val_score(
            model,
            X,
            y,
            cv=5,
            scoring="accuracy"
        )

        print("\nCross Validation Scores:")
        print(scores)

        print("\nAverage Cross-Validation Accuracy:")
        print(round(scores.mean() * 100, 2), "%")

        # Step 10: Train the model
        model.fit(X_train, y_train)

        # Step 11: Test the model
        test_accuracy = model.score(X_test, y_test)

        print("\nTest Accuracy:")
        print(round(test_accuracy * 100, 2), "%")

        # Step 12: Predict results
        predictions = model.predict(X_test)

        print("\nActual Results:")
        print(y_test.tolist())

        print("\nPredicted Results:")
        print(predictions.tolist())
