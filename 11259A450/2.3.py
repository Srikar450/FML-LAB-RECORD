
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the dataset
data = pd.read_csv("data.csv", sep="\t", encoding="utf-8-sig")

# Step 2: Clean column names
data.columns = data.columns.astype(str).str.strip()

# Step 3: Display available columns
print("Available Columns:", data.columns.tolist())

# Step 4: Check required columns
required_columns = ["Hours_Studied", "Marks"]

if not all(col in data.columns for col in required_columns):
    print("Error: Required columns not found.")
    print("Required columns:", required_columns)

else:
    # Step 5: Select required columns
    data = data[required_columns].copy()

    # Step 6: Convert values to numbers
    for col in required_columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # Step 7: Remove missing or invalid values
    data = data.dropna()

    # Step 8: Prepare input and output
    X = data[["Hours_Studied"]]
    y = data["Marks"]

    if len(data) < 5 or X["Hours_Studied"].nunique() < 2:
        print("Error: Need at least 5 valid rows and different study hours.")

    else:
        # Step 9: Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        # Step 10: Create and train the model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Step 11: Predict test marks
        y_pred = model.predict(X_test)

        # Step 12: Display results
        print("\n--- Linear Regression Results ---")
        print("Training Samples:", len(X_train))
        print("Testing Samples:", len(X_test))
        print("Slope:", round(model.coef_[0], 2))
        print("Intercept:", round(model.intercept_, 2))

        print("\nActual Marks:", y_test.tolist())
        print("Predicted Marks:", y_pred.round(2).tolist())

        print("\nMean Squared Error:",
              round(mean_squared_error(y_test, y_pred), 2))

        if len(y_test) >= 2:
            print("R2 Score:", round(r2_score(y_test, y_pred), 2))
        else:
            print("R2 Score: Not available; need at least 2 test samples.")

        # Step 13: Predict marks for new study hours
        try:
            hours = float(input("\nEnter hours studied: "))

            new_data = pd.DataFrame({
                "Hours_Studied": [hours]
            })

            prediction = model.predict(new_data)
            print("Predicted Marks:", round(prediction[0], 2))

        except ValueError:
            print("Please enter a valid number.")

        # Step 14: Plot the regression line
        sorted_data = X.sort_values("Hours_Studied")

        plt.scatter(
            X["Hours_Studied"],
            y,
            label="Actual Data"
        )

        plt.plot(
            sorted_data["Hours_Studied"],
            model.predict(sorted_data),
            color="red",
            label="Regression Line"
        )

        plt.xlabel("Hours Studied")
        plt.ylabel("Marks")
        plt.title("Linear Regression: Study Hours vs Marks")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
