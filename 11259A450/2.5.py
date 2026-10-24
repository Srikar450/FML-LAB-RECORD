
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

try:
    # Step 1: Load the tab-separated dataset
    data = pd.read_csv("data.csv", sep="\t", encoding="utf-8-sig")

    # Step 2: Clean column names
    data.columns = data.columns.astype(str).str.strip()

    # Step 3: Check required columns
    required_columns = [
        "Hours_Studied",
        "Attendance",
        "Assignments",
        "Result"
    ]

    print("Available Columns:", data.columns.tolist())

    missing = [
        col for col in required_columns
        if col not in data.columns
    ]

    if missing:
        print("Missing columns:", missing)

    else:
        # Step 4: Select required columns
        data = data[required_columns].copy()

        features = [
            "Hours_Studied",
            "Attendance",
            "Assignments"
        ]

        # Step 5: Convert features to numeric values
        for col in features:
            data[col] = pd.to_numeric(data[col], errors="coerce")

        # Clean target values
        data["Result"] = data["Result"].astype(str).str.strip()

        # Step 6: Remove missing values
        data = data.dropna(subset=required_columns)
        data = data[data["Result"] != ""]

        X = data[features]
        y = data["Result"]

        # Step 7: Check dataset
        if len(data) < 5 or y.nunique() < 2:
            print("Error: Need at least 5 valid rows and 2 Result classes.")

        else:
            # Step 8: Split into training and testing data
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42
            )

            # Step 9: Create SVM model with feature scaling
            model = make_pipeline(
                StandardScaler(),
                SVC(kernel="linear")
            )

            # Step 10: Train the model
            model.fit(X_train, y_train)

            # Step 11: Predict test results
            y_pred = model.predict(X_test)

            # Step 12: Display results
            print("\n--- SVM Classification Results ---")
            print("Training Samples:", len(X_train))
            print("Testing Samples:", len(X_test))

            print("\nActual Results:")
            print(y_test.tolist())

            print("\nPredicted Results:")
            print(y_pred.tolist())

            print("\nAccuracy:",
                  round(accuracy_score(y_test, y_pred) * 100, 2), "%")

            print("\nConfusion Matrix:")
            print(confusion_matrix(
                y_test,
                y_pred,
                labels=model.classes_
            ))

            print("\nClassification Report:")
            print(classification_report(
                y_test,
                y_pred,
                labels=model.classes_,
                zero_division=0
            ))

except FileNotFoundError:
    print("Error: data.csv was not found. Keep it in your program folder.")

except pd.errors.EmptyDataError:
    print("Error: data.csv is empty.")

except Exception as e:
    print("Error:", e)
