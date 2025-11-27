import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm   

def simple_regressions(df, variables, target="systolic_bp"):
    rows = []
    models = {} 
    
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
        models[var] = model

    return pd.DataFrame(rows), models


def plot_scatter_with_regression(df, var, target="systolic_bp", ax=None, model=None):
    if ax is None:
        fig, ax = plt.subplots()

    X = df[[var]]
    y = df[target]

    # Om ingen modell skickas in → traina direkt (flexiblare)
    if model is None:
        model = LinearRegression().fit(X, y)

    # Scatter
    ax.scatter(df[var], df[target], alpha=0.6, edgecolor="black")

    # Prediktion
    x_vals = np.linspace(df[var].min(), df[var].max(), 200)
    x_vals_df = pd.DataFrame({var: x_vals})
    y_vals = model.predict(x_vals_df)

    ax.plot(x_vals, y_vals, color="red", linewidth=2)

    R2 = model.score(X, y)
    ax.text(0.05, 0.95, f"R² = {R2:.2f}",
            transform=ax.transAxes, fontsize=12,
            bbox=dict(facecolor="white", alpha=0.6))

    ax.set_xlabel(var)
    ax.set_ylabel(target)
    ax.set_title(f"{target} vs {var} — regression")
    ax.grid(alpha=0.3)


def regression_panel(df, var, target="systolic_bp", model=None):
    """
    Skapar en 2x2-panel för vald variabel:
    1) Scatter + regression
    2) Residualplot
    3) Histogram + normalfördelning
    4) Q-Q-plot
    """

    # --- Modell ---
    X = df[[var]]
    y = df[target]

    if model is None:
        model = LinearRegression().fit(X, y)

    y_pred = model.predict(X)
    residuals = y - y_pred

    fig, axes = plt.subplots(2,2,figsize=(12,9))

    # ================= 1 SCATTER + REGRESSION =================
    ax = axes[0,0]
    x_grid = np.linspace(df[var].min(), df[var].max(), 200)
    pred_df = pd.DataFrame({var: x_grid})
    pred_y = model.predict(pred_df)

    ax.scatter(df[var], y, alpha=0.6, edgecolor="k")
    ax.plot(x_grid, pred_y, color="red", linewidth=2)
    
    R2 = model.score(X, y)
    ax.text(0.02, 0.98, f"R² = {R2:.2f}",
            transform=ax.transAxes, fontsize=12,
            va="top", ha="left",
            bbox=dict(facecolor="white", alpha=0.6))
    
    ax.set_title("Regression: Scatter + linje")
    ax.set_xlabel(var)
    ax.set_ylabel(target)
    ax.grid(alpha=0.3)

    # ================= 2 RESIDUALPLOT =================
    ax = axes[0,1]
    ax.scatter(y_pred, residuals, edgecolor="k", alpha=0.6)
    ax.axhline(0, color="red", linewidth=2)
    ax.set_title("Residualer vs Prediktion")
    ax.set_xlabel("Prediktion")
    ax.set_ylabel("Residual")
    ax.grid(alpha=0.3)

    # ================= 3 HISTOGRAM + NORMALKURVA =================
    ax = axes[1,0]
    ax.hist(residuals, bins=25, density=True, alpha=0.6, edgecolor="k")
    μ, σ = residuals.mean(), residuals.std()
    xs = np.linspace(residuals.min(), residuals.max(), 200)
    ax.plot(xs, norm.pdf(xs, μ, σ), color="darkorange", linewidth=2)
    ax.set_title("Histogram + Normalfördelning")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Täthet")
    ax.grid(alpha=0.3)

    # ================= 4 Q-Q-PLOT =================
    ax = axes[1,1]
    sm.qqplot(residuals, line="45", ax=ax)
    ax.set_title("Q–Q plot residualer")
    ax.grid(alpha=0.3)

    # ================= HUVUDTITEL =================
    fig.suptitle(f"Regressionsdiagnostik — {target} ~ {var}", fontsize=18, weight="bold")
    plt.tight_layout(rect=[0,0,1,0.95])  # ger plats för huvudrubrik
    plt.show()


def regression_diagnostics(df, var, target="systolic_bp", model=None):
    """
    Plottar 3 figurer i en rad för en variabel:
    1) Scatter + regression
    2) Residualplot
    3) Q-Q plot
    """

    # --- Hämta y & X ---
    X = df[[var]]
    y = df[target]

    # --- Återanvänd eller beräkna modell ---
    if model is None:
        model = LinearRegression().fit(X, y)

    y_pred = model.predict(X)
    residuals = y - y_pred

    # --- Subplots (1 rad, 3 kol) ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ==================================================
    # 1. Scatter + regressionslinje
    # ==================================================
    regression_x = np.linspace(df[var].min(), df[var].max(), 200)
    reg_df = pd.DataFrame({var: regression_x})
    reg_y = model.predict(reg_df)

    axes[0].scatter(df[var], y, edgecolor="black", alpha=0.6)
    axes[0].plot(regression_x, reg_y, color="red", linewidth=2)
    axes[0].set_title(f"{target} ~ {var}")

    # ==================================================
    # 2. Residualplot
    # ==================================================
    axes[1].scatter(y_pred, residuals, alpha=0.6, edgecolor="black")
    axes[1].axhline(0, color="red", linewidth=2)
    axes[1].set_title("Residualer vs Prediktion")
    axes[1].set_xlabel("Prediktion")
    axes[1].set_ylabel("Residual")

    # ==================================================
    # 3. Q-Q plot
    # ==================================================
    sm.qqplot(residuals, line="45", ax=axes[2])
    axes[2].set_title("Q–Q plot residualer")

    plt.tight_layout()
    plt.show()


def qq_diagnostics(df, var, target="systolic_bp"):
    """
    Visar residualernas fördelning för enkel linjär regression:
      - Histogram + normalfördelningskurva
      - Förbättrad Q–Q-plot
    """

    # 1. Anpassa enkel linjär regression
    X = df[[var]]
    y = df[target]
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    residuals = y - y_pred

    # 2. Figur: 1 rad, 2 kolumner
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ------------------------------------
    # Vänster: histogram + normal-kurva
    # ------------------------------------
    ax = axes[0]
    ax.hist(residuals, bins=30, density=True, alpha=0.6, edgecolor="black")

    # Teoretisk normalfördelning med samma medelvärde & std som residualerna
    mu, sigma = residuals.mean(), residuals.std()
    xs = np.linspace(residuals.min(), residuals.max(), 200)
    ax.plot(xs, norm.pdf(xs, mu, sigma), linewidth=2)

    ax.set_title(f"Residualfördelning — {target} ~ {var}")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Täthet")
    ax.grid(alpha=0.3)

    # ------------------------------------
    # Höger: förbättrad Q–Q-plot
    # ------------------------------------
    ax = axes[1]
    sm.qqplot(residuals, line="45", ax=ax)
    ax.set_title("Q–Q-plot residualer")
    ax.grid(alpha=0.3)
    ax.set_aspect("equal", "box")  # gör linjen 45° visuellt korrekt

    plt.tight_layout()
    plt.show()



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