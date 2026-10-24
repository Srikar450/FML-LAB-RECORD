
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
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
    features = ["Hours_Studied", "Attendance", "Assignments"]
    required_columns = features + ["Result"]

    print("Available Columns:", data.columns.tolist())

    missing = [col for col in required_columns if col not in data.columns]

    if missing:
        print("Missing columns:", missing)

    else:
        # Step 4: Select required columns
        data = data[required_columns].copy()

        # Step 5: Convert input features to numeric values
        for col in features:
            data[col] = pd.to_numeric(data[col], errors="coerce")

        # Step 6: Clean target values and remove invalid rows
        data["Result"] = data["Result"].astype(str).str.strip()
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

            # Step 9: Create AdaBoost model
            model = AdaBoostClassifier(
                n_estimators=50,
                random_state=42
            )

            # Step 10: Train the model
            model.fit(X_train, y_train)

            # Step 11: Predict results
            y_pred = model.predict(X_test)

            # Step 12: Display results
            print("\n--- AdaBoost Classification Results ---")
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

            # Step 13: Plot feature importance
            importance = model.feature_importances_

            plt.figure(figsize=(8, 5))
            plt.bar(features, importance)
            plt.xlabel("Features")
            plt.ylabel("Importance")
            plt.title("AdaBoost Feature Importance")
            plt.xticks(rotation=15)
            plt.tight_layout()
            plt.show()

except FileNotFoundError:
    print("Error: data.csv was not found. Keep it in your program folder.")

except pd.errors.EmptyDataError:
    print("Error: data.csv is empty.")

except Exception as e:
    print("Error:", e)
