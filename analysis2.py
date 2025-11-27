import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm  

def _fit_simple_regression(df, var, target="systolic_bp", model=None):
    """
    Helper to fit simple linear regression for one predictor.

    Returns X, y, model, predictions, and residuals for reuse in diagnostics.
    """
    X = df[[var]]
    y = df[target]

    if model is None:
        model = LinearRegression().fit(X, y)

    y_pred = model.predict(X)
    residuals = y - y_pred

    return X, y, model, y_pred, residuals



def simple_regressions(df, variables, target="systolic_bp"):
    """
    Fit separate simple linear regressions for each variable in `variables`.

    Returns
    -------
    summary_df : pd.DataFrame
        One row per predictor with intercept, coefficient, and R².
    models : dict
        Mapping {variable_name: fitted LinearRegression model}.
    """

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
    """
    Plot scatter + regression line for a single predictor vs target.

    If a pre-fitted model is provided it is reused, otherwise a new
    simple LinearRegression is fit internally.
    """
    X, y, model, y_pred, residuals = _fit_simple_regression(df, var, target, model)

    if ax is None:
        fig, ax = plt.subplots()

    # Scatter
    ax.scatter(df[var], y, alpha=0.6, edgecolor="black")

    # Regressionslinje
    x_vals = np.linspace(df[var].min(), df[var].max(), 200)
    x_vals_df = pd.DataFrame({var: x_vals})
    y_vals = model.predict(x_vals_df)
    ax.plot(x_vals, y_vals, color="red", linewidth=2)

    # R²
    R2 = model.score(X, y)
    ax.text(
        0.05, 0.95,
        f"R² = {R2:.3f}",
        transform=ax.transAxes,
        fontsize=12,
        ha="left", va="top",
        bbox=dict(facecolor="white", alpha=0.6)
    )

    ax.set_xlabel(var)
    ax.set_ylabel(target)
    ax.set_title(f"{target} vs {var} — regression")
    ax.grid(alpha=0.3)


def residual_plot(df, var, target="systolic_bp", ax=None, model=None):
    """
    Residuals vs predicted values for a simple linear regression.

    Useful to assess linearity and homoscedasticity assumptions.
    """
    X, y, model, y_pred, residuals = _fit_simple_regression(df, var, target, model)

    if ax is None:
        fig, ax = plt.subplots()

    residual_vs_pred_from_arrays(ax, y_pred, residuals)


def residual_hist(df, var, target="systolic_bp", ax=None, model=None):
    """
    Histogram + normal curve for residuals of a simple regression model.
    """
    X, y, model, y_pred, residuals = _fit_simple_regression(df, var, target, model)

    if ax is None:
        fig, ax = plt.subplots()

    residual_hist_from_array(ax, residuals, title="Histogram + Normalfördelning")


def residual_qq(df, var, target="systolic_bp", ax=None, model=None):
    """
    Q–Q plot of residuals from a simple regression model.

    Used to visually check approximate normality of residuals.
    """
    X, y, model, y_pred, residuals = _fit_simple_regression(df, var, target, model)

    if ax is None:
        fig, ax = plt.subplots()

    residual_qq_from_array(ax, residuals, title="Q–Q plot residualer")


def regression_panel(df, var, target="systolic_bp", model=None):
    """
    Create a 2×2 diagnostic panel for a simple regression: 
    scatter+line, residual vs predicted, histogram, and Q–Q plot.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # 1) Scatter + regression
    plot_scatter_with_regression(
        df, var, target=target, ax=axes[0,0], model=model
    )

    # 2) Residualplot
    residual_plot(
        df, var, target=target, ax=axes[0,1], model=model
    )

    # 3) Histogram + normalfördelning
    residual_hist(
        df, var, target=target, ax=axes[1,0], model=model
    )

    # 4) Q–Q-plot
    residual_qq(
        df, var, target=target, ax=axes[1,1], model=model
    )

    fig.suptitle(
        f"Regressionsdiagnostik — {target} ~ {var}",
        fontsize=18, weight="bold"
    )
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.show()


# ====== GENERELLA HJÄLPPLOTTAR FÖR RESIDUALER (FRÅN ARRAYER) ======

def residual_vs_pred_from_arrays(ax, y_pred, residuals):
    """Plot residuals vs predictions, given ready-made arrays."""
    ax.scatter(y_pred, residuals, edgecolor="k", alpha=0.6)
    ax.axhline(0, color="red", linewidth=2)
    ax.set_title("Residualer vs Prediktion")
    ax.set_xlabel("Prediktion")
    ax.set_ylabel("Residual")
    ax.grid(alpha=0.3)


def residual_hist_from_array(ax, residuals, title="Histogram + Normalfördelning"):
    """Plot histogram + normal curve for a given residual array."""
    ax.hist(residuals, bins=25, density=True, alpha=0.6, edgecolor="k")
    μ, σ = residuals.mean(), residuals.std()
    xs = np.linspace(residuals.min(), residuals.max(), 200)
    ax.plot(xs, norm.pdf(xs, μ, σ), color="darkorange", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Residual")
    ax.set_ylabel("Täthet")
    ax.grid(alpha=0.3)


def residual_qq_from_array(ax, residuals, title="Q–Q residualer"):
    """Q–Q plot for an array of residuals (normality diagnostic)."""
    sm.qqplot(residuals, line="45", ax=ax)
    ax.set_title(title)
    ax.grid(alpha=0.3)


def plot_partial_effect(ax, df, var_focus, var_hold, target, model, vars2):
    """
    Plot the partial effect of `var_focus` in a multiple regression.

    Holds `var_hold` fixed at its mean to show the adjusted relationship
    between var_focus and the target.
    """

    y = df[target].values

    # mappa koefficienter till variabelnamn
    coef_map = dict(zip(vars2, model.coef_))
    b0 = model.intercept_
    b_focus = coef_map[var_focus]
    b_hold = coef_map[var_hold]

    hold_mean = df[var_hold].mean()
    grid = np.linspace(df[var_focus].min(), df[var_focus].max(), 200)

    ax.scatter(df[var_focus], y, alpha=0.5, edgecolor="k")
    ax.plot(grid, b0 + b_focus * grid + b_hold * hold_mean, linewidth=2)

    ax.set_title(f"Partiell effekt av {var_focus} ( {var_hold} = medel )")
    ax.set_xlabel(var_focus)
    ax.set_ylabel(target)
    ax.grid(alpha=0.3)


# ====== FULL PANEL FÖR MULTIPEL REGRESSION (TVÅ PREDIKTORER) ======

def multiple_regression_panel_full(df, vars2=None, target="systolic_bp"):
    """
    Full diagnostic panel for a two-predictor multiple regression.

    Includes:
      - Partial effect plots for each predictor
      - Residual vs prediction plot
      - Residual histogram + normal curve
      - Residual Q–Q plot
      - Summary text box with R², adjusted R², and coefficients
    """

    if vars2 is None:
        vars2 = ['age', 'cholesterol']

    v1, v2 = vars2

    # --- Anpassa modell ---
    X = df[vars2]
    y = df[target]
    model = LinearRegression().fit(X, y)

    y_pred = model.predict(X)
    residuals = y - y_pred

    # --- R² och Adjusted R² ---
    R2 = model.score(X, y)
    n = len(df)
    p = len(vars2)
    adj_R2 = 1 - (1 - R2) * (n - 1) / (n - p - 1)

    # --- Plottar ---
    fig, axes = plt.subplots(3, 2, figsize=(14, 16))

    # 1) Partiell effekt av v1
    plot_partial_effect(
        axes[0, 0],
        df=df,
        var_focus=v1,
        var_hold=v2,
        target=target,
        model=model,
        vars2=vars2,
    )

    # 2) Partiell effekt av v2
    plot_partial_effect(
        axes[0, 1],
        df=df,
        var_focus=v2,
        var_hold=v1,
        target=target,
        model=model,
        vars2=vars2,
    )

    # 3) Residual vs prediktion
    residual_vs_pred_from_arrays(axes[1, 0], y_pred, residuals)

    # 4) Histogram av residualer
    residual_hist_from_array(axes[1, 1], residuals, title="Residualdistribution")

    # 5) Q–Q-plot av residualer
    residual_qq_from_array(axes[2, 0], residuals, title="Q–Q residualer")

    # 6) Koef + R²-ruta
    ax = axes[2, 1]
    ax.axis('off')

    coef_map = dict(zip(vars2, model.coef_))
    text = (
        f"R²        = {R2:.3f}\n"
        f"Adj R²    = {adj_R2:.3f}\n\n"
        f"Intercept = {model.intercept_:.2f}\n"
        f"{v1} koef  = {coef_map[v1]:.2f}\n"
        f"{v2} koef  = {coef_map[v2]:.2f}\n"
    )
    ax.text(0.05, 0.8, text,
            fontsize=14, va="top", family="monospace")

    fig.suptitle(
        f"Multipel regressionsdiagnostik — {target} ~ {v1} + {v2}",
        fontsize=18, fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def ols_full(df, variables, target="systolic_bp"):
    """
    Fit a full OLS model using statsmodels for the given predictors.

    Returns
    -------
    statsmodels.regression.linear_model.RegressionResultsWrapper
        Fitted OLS model with full summary capabilities.
    """
    X = sm.add_constant(df[variables])
    y = df[target]
    model = sm.OLS(y, X).fit()
    return model


def compare_models(summary_df, model_two, model_full, model_reduced):
    """
    Build a comparison table of R² and comments for several regression models.

    Expects:
      - simple models in `summary_df`
      - a two-predictor model (model_two)
      - a full model (model_full)
      - a reduced model (model_reduced)
    """

    summary_models = pd.DataFrame({
        "Modell": [
            "Age (enkel)", 
            "Cholesterol (enkel)", 
            "Weight (enkel)", 
            "Height (enkel)",
            "Age + Cholesterol (multipel)",
            "Full modell (alla 4 variabler)",
            "Reducerad modell (Age + Weight)"
        ],
        "R²": [
            summary_df.loc[summary_df.Variable=="age", "R²"].values[0],
            summary_df.loc[summary_df.Variable=="cholesterol", "R²"].values[0],
            summary_df.loc[summary_df.Variable=="weight", "R²"].values[0],
            summary_df.loc[summary_df.Variable=="height", "R²"].values[0],
            model_two.rsquared,  
            model_full.rsquared,
            model_reduced.rsquared
        ],
        "Kommentar": [
            "Starkast enskilt samband",
            "Tydligt svagare",
            "Mycket svagt samband",
            "Ingen effekt",
            "Liten förbättring jämfört med age ensam",
            "Högst R² men marginell förbättring",
            "Nästan lika bra som full modell → föredras"
        ]
    })

    return summary_models  # tabellen kan visas med display()