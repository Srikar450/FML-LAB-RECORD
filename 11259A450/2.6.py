
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

try:
    # Step 1: Load the tab-separated dataset
    data = pd.read_csv("data.csv", sep="\t", encoding="utf-8-sig")

    # Step 2: Clean column names
    data.columns = data.columns.astype(str).str.strip()

    print("Available Columns:", data.columns.tolist())

    # Step 3: Check required columns
    required_columns = ["Hours_Studied", "Marks"]

    missing = [
        col for col in required_columns
        if col not in data.columns
    ]

    if missing:
        print("Missing columns:", missing)

    else:
        # Step 4: Select required columns
        X = data[required_columns].copy()

        # Step 5: Convert values to numbers
        for col in required_columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

        # Step 6: Remove missing values
        valid_rows = X.dropna().index
        X = X.loc[valid_rows].copy()
        data = data.loc[valid_rows].copy()

        # Step 7: Check dataset size
        if len(X) < 3:
            print("Error: At least 3 valid rows are needed.")

        else:
            # Step 8: Scale the features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Step 9: Create and train K-Means model
            kmeans = KMeans(
                n_clusters=3,
                random_state=42,
                n_init=10
            )

            clusters = kmeans.fit_predict(X_scaled)

            # Step 10: Add cluster labels
            data["Cluster"] = clusters

            print("\n--- K-Means Clustering Results ---")
            print(data.to_string(index=False))

            # Step 11: Convert centroids back to original units
            centroids = scaler.inverse_transform(
                kmeans.cluster_centers_
            )

            # Step 12: Plot clusters and centroids
            plt.scatter(
                X["Hours_Studied"],
                X["Marks"],
                c=clusters,
                cmap="viridis",
                s=70,
                label="Data Points"
            )

            plt.scatter(
                centroids[:, 0],
                centroids[:, 1],
                marker="X",
                s=250,
                color="red",
                label="Centroids"
            )

            plt.xlabel("Hours Studied")
            plt.ylabel("Marks")
            plt.title("K-Means Clustering")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

except FileNotFoundError:
    print("Error: data.csv was not found. Keep it in your program folder.")

except pd.errors.EmptyDataError:
    print("Error: data.csv is empty.")

except Exception as e:
    print("Error:", e)
