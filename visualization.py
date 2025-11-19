import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import pandas as pd


NUM_COLS = ["age", "height", "weight", "systolic_bp", "cholesterol"]
CAT_COLS = ["sex", "smoker", "disease"]


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate summary statistics (mean, median, std, min, max)
    for predefined numerical columns.
    """
    summary = df[NUM_COLS].agg(["mean", "median", "std", "min", "max"]).T
    return summary


def plot_hist(data, col, ax, bins=30, color="skyblue"):
    """
    Draw a histogram for a column.
    """
    ax.hist(data, bins=bins, color=color, edgecolor="black", alpha=0.7)
    ax.set_title(f"Distribution of {col}", fontsize=12)
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    ax.grid(axis="both", alpha=0.5)


def plot_box(data, col, ax, color="lightblue", position=1):
    """
    Draw a horizontal boxplot for a column.
    """
    kwargs = dict(
        vert=False, patch_artist=True,
        boxprops=dict(facecolor=color, color="black"),
        medianprops=dict(color="red"),
    )
    if position is not None:
        kwargs["positions"] = [position]

    ax.boxplot(data, **kwargs)
    ax.set_xlabel(col)
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.5)


def plot_group_box(df, value_col, group_col, ax,
                   order=None, colors=None, title=None):
    """
    Draw boxplots for value_col grouped by group_col on the same axis.
    """
    groups = list(df[group_col].dropna().unique()) if order is None else order
    default_colors = {"M": "lightblue", "F": "lightcoral"}

    plotted = []
    for pos, group in enumerate(groups, start=1):
        data = df.loc[df[group_col] == group, value_col].dropna()
        if data.empty:
            continue
        color = default_colors.get(group, "lightgray") if colors is None else colors.get(group, "lightgray")
        plot_box(data, value_col, ax, color=color, position=pos)
        plotted.append((pos, group))

    if plotted:
        positions, labels = zip(*plotted)
        ax.set_yticks(positions)
        ax.set_yticklabels(labels)
    else:
        ax.set_yticks([])

    ax.set_title(title or f"{value_col} by {group_col}", fontsize=14)


# def plot_proportion_bar(df, col, ax, title=None, colors=None, order=None):
#     """
#     Draw a bar chart of proportions (%) for a categorical variable.
#     """
#     prop = df[col].value_counts(normalize=True) * 100

#     if order is not None:
#         prop = prop.reindex(order).fillna(0)

#         bar_colors = [colors.get(category, "lightgray") for category in prop.index] if colors else "lightblue"
#     else:
#         bar_colors = colors if colors is not None else "lightblue"

#     ax.bar(prop.index, prop.values, color=bar_colors, edgecolor="black")

#     ax.set_ylabel("Andel (%)")
#     ax.set_xlabel(col)
#     ax.set_title(title or f"Andel per kategori i {col}")
#     ax.grid(axis="y", alpha=0.4)

#     for i, v in enumerate(prop.values):
#         ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=10)


def plot_proportion_bar(df, col, ax, title=None, colors=None, order=None):
    """
    Draw a bar chart of proportions (%) for a categorical variable.
    Automatically handles nice labels and colors for sex, smoker, and disease.
    """

    prop = df[col].value_counts(normalize=True) * 100

    if order is not None:
        prop = prop.reindex(order).fillna(0)

    categories = list(prop.index)
    positions = np.arange(len(categories))

    # --- Default label mapping ---
    label_map = {
        True: "Diseased",
        False: "Healthy",
        "Yes": "Smoker",
        "No": "Non-smoker",
        "M": "Male",
        "F": "Female"
    }

    tick_labels = [label_map.get(cat, str(cat)) for cat in categories]

    # --- Default color mapping ---
    default_colors = {
        "Male": "lightblue",
        "Female": "lightpink",
        "Smoker": "salmon",
        "Non-smoker": "lightgreen",
        "Diseased": "orange",
        "Healthy": "lightgray",
    }

    bar_colors = [default_colors.get(lbl, "lightblue") for lbl in tick_labels]

    # If user passed a custom color dict → override defaults
    if colors is not None and isinstance(colors, dict):
        bar_colors = [colors.get(lbl, c) for lbl, c in zip(tick_labels, bar_colors)]

    # Draw bars
    ax.bar(positions, prop.values, color=bar_colors, edgecolor="black")

    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels)

    ax.set_ylabel("Percentage (%)")
    ax.set_xlabel(col)
    ax.set_title(title or f"Distribution of {col}")
    ax.grid(axis="y", alpha=0.4)

    # Show % above bars
    for pos, v in zip(positions, prop.values):
        ax.text(pos, v + 1, f"{v:.1f}%", ha="center", fontsize=10)





def plot_all_num_distributions(df: pd.DataFrame, cols=None):
    """
    Create combined histogram + boxplot for a set of numerical columns.
    """
    if cols is None:
        cols = NUM_COLS

    n_cols = 2
    n_rows = int(np.ceil(len(cols) / n_cols))

    fig = plt.figure(figsize=(12, 6 * n_rows))
    outer = fig.add_gridspec(n_rows, n_cols, hspace=0.35, wspace=0.25)

    for i, col in enumerate(cols):
        row_gr = i // n_cols
        c = i % n_cols

        inner = outer[row_gr, c].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.0)

        ax_hist = fig.add_subplot(inner[0])
        ax_box = fig.add_subplot(inner[1], sharex=ax_hist)

        data = df[col].dropna()
        plot_hist(data, col, ax_hist)
        plot_box(data, col, ax_box)

        ax_hist.tick_params(axis="x", which="both", labelbottom=False, bottom=False)
        ax_hist.spines["bottom"].set_visible(False)

        ax_hist.set_title(f"Distribution and boxplot for {col}", fontsize=14)

    plt.show()


def plot_all_categorical_bars(df, cols=None):
    """
    Create bar charts showing the proportion (%) of categorical variables.
    """

    # Om inga kolumner skickas in → använd CAT_COLS
    if cols is None:
        cols = CAT_COLS   # ← använder din egen lista

    # Define readable titles for each column
    pretty_titles = {
        "sex": "Sex distribution (%)",
        "smoker": "Smoking status (%)",
        "disease": "Disease status (%)"
    }

    n_cols = 2
    n_rows = int(np.ceil(len(cols) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axes = axes.flatten()

    for ax, col in zip(axes, cols):

        unique_vals = df[col].dropna().unique()
        colors = {val: "lightblue" for val in unique_vals}

        plot_proportion_bar(
            df=df,
            col=col,
            ax=ax,
            title=pretty_titles.get(col, f"{col} (%)"),
            colors=colors,
            order=unique_vals
        )

    # ta bort tomma rutor om antal inte matchar grid
    for i in range(len(cols), len(axes)):
        fig.delaxes(axes[i])

    fig.tight_layout()
    plt.show()
    return fig
