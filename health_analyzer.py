# health_analyzer.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from visualization import plot_hist, get_category_color


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
            boxprops=dict(color="black"),
            medianprops=dict(color="red"),
        )
        
        # färglägg boxarna enligt CATEGORY_COLORS
        box_colors = [
            get_category_color(group_col, g, default="lightblue")
            for g in order
        ]
        for patch, c in zip(bp["boxes"], box_colors):
            patch.set_facecolor(c)

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
    

    def plot_bp_hist(self, bins=30):
        """
        Plottar histogram för systoliskt blodtryck.
        """
        values = self.df[self.bp_col].dropna()  # 1D-array med bara systoliskt BT
        
        fig, ax = plt.subplots(figsize=(6, 3))
        plot_hist(values, self.bp_col, ax=ax, bins=bins)
        plt.show()


    def plot_bp_hist_by_group(self, group_col, bins=30):
        """
        Ritar histogram för systoliskt blodtryck uppdelat på grupper.
        """
        groups = self.df[group_col].dropna().unique()

        fig, axes = plt.subplots(1, len(groups), figsize=(6 * len(groups), 4))

        if len(groups) == 1:
            axes = [axes]

        for ax, g in zip(axes, groups):
            subset = self.df.loc[self.df[group_col] == g, self.bp_col].dropna()
            plot_hist(subset, f"{self.bp_col} ({g})", ax=ax, bins=bins)

        plt.tight_layout()
        plt.show()
    
    def plot_bp_hist_overlaid(self, group_col, bins=30, ax=None, show=True):
        """
        Överlagrat histogram för systoliskt BT, uppdelat på group_col
        (t.ex. 'smoker' eller 'sex').

        Om ax är None skapas en ny figur, annars ritas i det givna ax-objektet.
        """
        data = self.df[[group_col, self.bp_col]].dropna()
        groups = data[group_col].unique()

        # gemensamt bin-intervall
        min_bp = data[self.bp_col].min()
        max_bp = data[self.bp_col].max()
        bin_edges = np.linspace(min_bp, max_bp, bins + 1)

        # skapa ax om det inte skickats in
        #created_fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))

        #colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

        for g in groups:
            subset = data.loc[data[group_col] == g, self.bp_col]
            color = get_category_color(group_col, g, default="tab:blue")

            ax.hist(
                subset,
                bins=bin_edges,
                alpha=0.5,
                density=False,
                label=f"{group_col}: {g}",
                edgecolor="black",
                color=color,
            )

        ax.set_title(f"Överlagrat histogram: {self.bp_col} per {group_col}")
        ax.set_xlabel(self.bp_col)
        ax.set_ylabel("Frekvens")
        ax.grid(alpha=0.3)
        ax.legend()
        plt.tight_layout()
    
        # returnera så att vi kan använda den om vi vill
        return ax


    def plot_bp_hist_box_grid(self, group_cols=None, bins=30):
        """
        Ritar en grid med 3 rader (eller fler) och 2 kolumner:
        - vänster: överlagrat histogram för systoliskt BT per grupp
        - höger : boxplot för systoliskt BT per grupp
        """
        if group_cols is None:
            group_cols = ["sex", "smoker", "disease"]

        n_rows = len(group_cols)
        fig, axes = plt.subplots(n_rows, 2, figsize=(12, 3 * n_rows), squeeze=False)

        for i, col in enumerate(group_cols):
            ax_hist = axes[i, 0]
            ax_box  = axes[i, 1]

            # överlagt histogram (använd befintligt ax, show=False)
            self.plot_bp_hist_overlaid(group_col=col, bins=bins, ax=ax_hist, show=False)

            # boxplot
            self.plot_bp_box_by_group(col, ax=ax_box,
                                    title=f"{self.bp_col} by {col}")

        fig.tight_layout()
        return fig, axes