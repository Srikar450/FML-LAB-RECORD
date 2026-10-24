
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

        # Step 5: Convert features to numbers
        for col in features:
            data[col] = pd.to_numeric(data[col], errors="coerce")

        # Step 6: Clean labels
        data["Result"] = data["Result"].astype(str).str.strip().str.title()

        # Convert Pass/Fail labels to numbers
        data["Result"] = data["Result"].map({
            "Fail": 0,
            "Pass": 1
        })

        # Remove missing or invalid rows
        data = data.dropna(subset=required_columns)

        X = data[features]
        y = data["Result"].astype(int)

        # Step 7: Check dataset
        if len(data) < 10 or y.nunique() < 2:
            print("Error: Need at least 10 valid rows containing Pass and Fail.")

        else:
            # Step 8: Split dataset
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )

            # Step 9: Scale features
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Step 10: Build neural network
            tf.random.set_seed(42)

            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(3,)),
                tf.keras.layers.Dense(8, activation="relu"),
                tf.keras.layers.Dense(4, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid")
            ])

            # Step 11: Compile model
            model.compile(
                optimizer="adam",
                loss="binary_crossentropy",
                metrics=["accuracy"]
            )

            # Step 12: Train model
            model.fit(
                X_train,
                y_train,
                epochs=50,
                batch_size=8,
                validation_split=0.2,
                verbose=1
            )

            # Step 13: Predict test results
            probabilities = model.predict(X_test, verbose=0).ravel()
            y_pred = (probabilities >= 0.5).astype(int)

            # Step 14: Display results
            print("\n--- Neural Network Results ---")
            print("Training Samples:", len(X_train))
            print("Testing Samples:", len(X_test))

            print("\nActual Results:")
            print(y_test.map({0: "Fail", 1: "Pass"}).tolist())

            print("\nPredicted Results:")
            print(pd.Series(y_pred).map({0: "Fail", 1: "Pass"}).tolist())

            print("\nAccuracy:",
                  round(accuracy_score(y_test, y_pred) * 100, 2), "%")

            print("\nConfusion Matrix:")
            print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

            print("\nClassification Report:")
            print(classification_report(
                y_test,
                y_pred,
                labels=[0, 1],
                target_names=["Fail", "Pass"],
                zero_division=0
            ))

except FileNotFoundError:
    print("Error: data.csv was not found. Keep it in your program folder.")

except pd.errors.EmptyDataError:
    print("Error: data.csv is empty.")

except ImportError:
    print("Error: Install the required libraries using:")
    print("pip install pandas scikit-learn tensorflow")

except Exception as e:
    print("Error:", e)
