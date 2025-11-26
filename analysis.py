import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

def simple_regressions(df, variables, target="systolic_bp"):
    rows = []
    for var in variables:
        X = df[[var]]
        y = df[target]
        model = LinearRegression().fit(X, y)
        rows.append({
            "Variable": var,
            "Intercept": model.intercept_,
            "Coefficient": model.coef_[0],
            "R²": model.score(X, y),
        })
    return pd.DataFrame(rows)


def plot_scatter_with_regression(
    ax,
    x: pd.Series,
    y: pd.Series,
    x_label: str,
    y_label: str,
    title: str | None = None,
):
    """
    Scatterplot + enkel linjär regressionslinje + R² i hörnet.
    """

    # Skapa DataFrame
    data = pd.DataFrame({"x": x, "y": y})

    # Scatter
    ax.scatter(data["x"], data["y"], alpha=0.6, edgecolor="black")

    # Enkel linjär regression
    model = LinearRegression().fit(data[["x"]], data["y"])

    # Grid för linjen
    x_vals = np.linspace(data["x"].min(), data["x"].max(), 200).reshape(-1, 1)
    x_vals_df = pd.DataFrame(x_vals, columns=["x"])   # viktigt för att slippa varning
    y_vals = model.predict(x_vals_df)

    ax.plot(x_vals, y_vals, color="red", linewidth=2)

    # R² i hörnet
    R2 = model.score(data[["x"]], data["y"])
    ax.text(
        0.05,
        0.95,
        f"R² = {R2:.2f}",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.6, edgecolor="gray"),
    )

    # Etiketter & titel
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.3)


def multiple_regression_two(df, vars2, target="systolic_bp"):
    y = df[target].values
    X = df[vars2].values
    model = LinearRegression().fit(X, y)
    return {
        "Intercept": model.intercept_,
        "Coefs": dict(zip(vars2, model.coef_)),
        "R²": model.score(X, y),
    }


def plot_partial_effects(df, vars2, target="systolic_bp"):
    """
    Ritar två partiella effekt-plots bredvid varandra (1 rad, 2 kolumner)
    för multipel regression med två variabler.
    """

    # === Hämta regressionsresultat via din funktion ===
    res = multiple_regression_two(df, vars2, target)
    v1, v2 = vars2
    b0 = res["Intercept"]
    b1 = res["Coefs"][v1]
    b2 = res["Coefs"][v2]

    X = df[vars2].values
    y = df[target].values

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))  # <-- 1 rad, 2 kolumner

    # ================================================
    # Plot 1 – Partial effekt av variabel 1
    v1_vals = np.linspace(df[v1].min(), df[v1].max(), 200)
    v2_mean = df[v2].mean()
    pred_v1 = b0 + b1 * v1_vals + b2 * v2_mean

    axes[0].scatter(df[v1], y, alpha=0.5, edgecolor="k")
    axes[0].plot(v1_vals, pred_v1, color="red", linewidth=2)
    axes[0].set_title(f"Partiell effekt av {v1} ( {v2} = medel )")
    axes[0].set_xlabel(v1)
    axes[0].set_ylabel(f"Prediktat {target}")
    axes[0].grid(alpha=0.3)

    # ================================================
    # Plot 2 – Partial effekt av variabel 2
    v2_vals = np.linspace(df[v2].min(), df[v2].max(), 200)
    v1_mean = df[v1].mean()
    pred_v2 = b0 + b1 * v1_mean + b2 * v2_vals

    axes[1].scatter(df[v2], y, alpha=0.5, edgecolor="k")
    axes[1].plot(v2_vals, pred_v2, color="blue", linewidth=2)
    axes[1].set_title(f"Partiell effekt av {v2} ( {v1} = medel )")
    axes[1].set_xlabel(v2)
    axes[1].set_ylabel(f"Prediktat {target}")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()



def ols_full(df, variables, target="systolic_bp"):
    X = sm.add_constant(df[variables])
    y = df[target]
    model = sm.OLS(y, X).fit()
    return model