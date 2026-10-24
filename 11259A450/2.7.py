
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
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
        X = X.dropna()
        data = data.loc[X.index].copy()

        # Step 7: Check dataset size
        if len(X) < 3:
            print("Error: At least 3 valid rows are required.")

        else:
            # Step 8: Scale the features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Step 9: Create linkage matrix
            linked = linkage(X_scaled, method="ward")

            # Step 10: Display dendrogram
            plt.figure(figsize=(10, 6))
            dendrogram(linked)

            plt.title("Dendrogram for Hierarchical Clustering")
            plt.xlabel("Data Points")
            plt.ylabel("Euclidean Distance")
            plt.tight_layout()
            plt.show()

            # Step 11: Perform hierarchical clustering
            hc = AgglomerativeClustering(n_clusters=3)
            clusters = hc.fit_predict(X_scaled)

            # Step 12: Add cluster labels
            data["Cluster"] = clusters

            print("\n--- Hierarchical Clustering Results ---")
            print(data.to_string(index=False))

            # Step 13: Plot the clusters
            plt.figure(figsize=(8, 5))

            plt.scatter(
                X["Hours_Studied"],
                X["Marks"],
                c=clusters,
                cmap="viridis",
                s=70
            )

            plt.xlabel("Hours Studied")
            plt.ylabel("Marks")
            plt.title("Hierarchical Clustering")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

except FileNotFoundError:
    print("Error: data.csv was not found. Keep it in your program folder.")

except pd.errors.EmptyDataError:
    print("Error: data.csv is empty.")

except Exception as e:
    print("Error:", e)
