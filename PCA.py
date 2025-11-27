import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def run_pca(df, variables, target="systolic_bp", n_components=2, plot=True):
    """
    Run PCA on selected variables and (optionally) visualize the first components.

    The variables are standardized before PCA. Returns a DataFrame with the
    principal components and the target, the loadings (variable contributions),
    the explained variance ratios, and the fitted PCA object.
    """

    # 1. Skala variabler
    X = df[variables].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. PCA-modell
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # 3. DataFrame för output
    pca_df = pd.DataFrame(
        X_pca, 
        columns=[f"PC{i+1}" for i in range(n_components)]
    )
    pca_df[target] = df[target].values

    # 4. Förklarad varians
    explained = pca.explained_variance_ratio_
    print("\nFörklarad varians per komponent:")
    for i, v in enumerate(explained, start=1):
        print(f" PC{i}: {v*100:.2f}%")
    print(f"Totalt: {explained.sum()*100:.2f}%")

    # 5. Loadings (variabelbidrag)
    loadings = pd.DataFrame(
        pca.components_.T,
        index=variables,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )
    print("\nLoadings:")
    print(loadings)

    # 6. Visualisering
    if plot and n_components >= 2:
        plt.figure(figsize=(8,6))
        scatter = plt.scatter(
            pca_df["PC1"], pca_df["PC2"],
            c=pca_df[target], cmap="coolwarm",
            edgecolor="k", alpha=0.75
        )
        plt.xlabel("PC1"); plt.ylabel("PC2")
        plt.title("PCA — komponentstruktur")
        cbar = plt.colorbar(scatter)
        cbar.set_label(target)
        plt.grid(alpha=0.3)
        plt.show()


    return pca_df, loadings, explained, pca
