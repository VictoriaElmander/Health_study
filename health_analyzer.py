# health_analyzer.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class HealthAnalyzer:
    """
    Class for analysing systolic blood pressure in groups.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe.
    bp_col : str, default "systolic_bp"
        Name of the systolic blood pressure column.
    """

    def __init__(self, df: pd.DataFrame, bp_col: str = "systolic_bp"):
        self.df = df.copy()
        self.bp_col = bp_col

    # -------------------------------------------------------
    # 1. Sammanfattning + CI per grupp
    # -------------------------------------------------------
    def bp_summary_by_group(self, group_col: str, confidence: float = 0.95) -> pd.DataFrame:
        """
        Calculate group-wise summary statistics and approx. confidence interval
        for mean systolic blood pressure.

        Returns a DataFrame with:
        n, mean, std, median, min, max, ci_lower, ci_upper.
        """
        data = self.df[[group_col, self.bp_col]].dropna()

        grp = data.groupby(group_col, observed=True)[self.bp_col]

        summary = grp.agg(
            n="count",
            mean="mean",
            std="std",
            median="median",
            min="min",
            max="max",
        )

        # Approximate normal CI: mean ± z * (std / sqrt(n))
        z = 1.96 if np.isclose(confidence, 0.95) else None
        if z is None:
            from scipy.stats import norm
            z = norm.ppf(0.5 + confidence / 2.0)

        summary["ci_lower"] = summary["mean"] - z * summary["std"] / np.sqrt(summary["n"])
        summary["ci_upper"] = summary["mean"] + z * summary["std"] / np.sqrt(summary["n"])

        # ordna kolumner och runda
        summary = summary[["n", "mean", "median", "std", "min", "max", "ci_lower", "ci_upper"]]
        summary = summary.round({"mean": 1, "median": 1, "std": 1, "min": 1, "max": 1,
                             "ci_lower": 1, "ci_upper": 1})


        return summary

    # -------------------------------------------------------
    # 2. Boxplot för systoliskt BT per grupp
    # -------------------------------------------------------
    def plot_bp_box_by_group(
        self,
        group_col: str,
        ax: plt.Axes | None = None,
        order=None,
        title: str | None = None,
    ):
        """
        Draw a horizontal boxplot of systolic BP split by a grouping variable
        (e.g. sex, smoker, disease).
        """
        data = self.df[[group_col, self.bp_col]].dropna()

        if order is None:
            order = list(data[group_col].dropna().unique())

        groups = []
        for g in order:
            vals = data.loc[data[group_col] == g, self.bp_col]
            if not vals.empty:
                groups.append(vals.values)
            else:
                groups.append(np.array([]))

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 3))

        bp = ax.boxplot(
            groups,
            vert=False,
            labels=order,
            patch_artist=True,
            boxprops=dict(facecolor="lightblue", color="black"),
            medianprops=dict(color="red"),
        )

        ax.set_xlabel(self.bp_col)
        ax.set_title(title or f"{self.bp_col} by {group_col}")
        ax.grid(axis="x", alpha=0.4)

        return ax

    # -------------------------------------------------------
    # 3. Multi-boxplot-grid för flera kategorier
    # -------------------------------------------------------
    def plot_bp_multi_boxgrid(self, group_cols=None):
        """
        Create a grid of boxplots for systolic BP by several categorical variables,
        e.g. ['sex', 'smoker', 'disease'].
        """
        if group_cols is None:
            # auto-välj kategoriska kolumner
            group_cols = [
                c for c in self.df.columns
                if str(self.df[c].dtype) in ("category", "bool", "object")
            ]

        n = len(group_cols)
        n_cols = 2
        n_rows = int(np.ceil(n / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows), squeeze=False)
        axes = axes.flatten()

        for ax, col in zip(axes, group_cols):
            self.plot_bp_box_by_group(col, ax=ax, title=f"{self.bp_col} by {col}")

        # ta bort tomma rutor om antalet inte delas exakt
        for j in range(len(group_cols), len(axes)):
            fig.delaxes(axes[j])

        fig.tight_layout()
        return fig, axes

    def analyze_bp_by_group(self, group_col: str, order=None, confidence: float = 0.95):
        """
        Skriv ut sammanfattande statistik + rita boxplot för systoliskt BT
        grupperat på en kategorisk variabel (t.ex. 'sex', 'smoker', 'disease').
        """
        summary = self.bp_summary_by_group(group_col, confidence=confidence)
        print(f"\nSystolic BP sammanfattning per {group_col}:")
        display(summary)

        fig, ax = plt.subplots(figsize=(6, 3))
        self.plot_bp_box_by_group(group_col, ax=ax, order=order,
                                  title=f"{self.bp_col} by {group_col}")
        plt.show()

        return summary