import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression

NUM_COLS = ["age", "height", "weight", "systolic_bp", "cholesterol"]
CAT_COLS = ["sex", "smoker", "disease"]

# -------------------------------------------------
# Gemensam färgpalett för kategoriska variabler
# -------------------------------------------------
CATEGORY_COLORS = {
    "sex": {
        "F": "lightpink",   # Female – ljusrosa
        "M": "lightblue",   # Male – ljusblå
    },
    "smoker": {
        "No":  "lightgreen",  # icke-rökare – blå
        "Yes": "salmon",  # rökare – sand/orange
    },
    "disease": {
        False: "lightgray",  # frisk – ljusgrå
        True:  "orange",  # sjuk – orange
    },
}


def get_category_color(var_name, level, default="#777777"):
    """
    Hämtar en konsekvent färg för en kategorinivå.
    Exempel: get_category_color("smoker", "Yes").
    """
    mapping = CATEGORY_COLORS.get(var_name, {})
    return mapping.get(level, default)


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


def plot_proportion_bar(df, col, ax, title=None, colors=None, order=None):
    """
    Draw a bar chart of proportions (%) for a categorical variable.
    Automatically handles nice labels and consistent colors.
    """

    # Calculate proportions
    prop = df[col].value_counts(normalize=True) * 100

    # Optional forced ordering
    if order is not None:
        prop = prop.reindex(order).fillna(0)

    categories = list(prop.index)
    positions = np.arange(len(categories))

    # --- Label mapping ---
    label_map = {
        True: "Diseased",
        False: "Healthy",
        "Yes": "Smoker",
        "No": "Non-smoker",
        "M": "Male",
        "F": "Female",
    }
    xlabels = [label_map.get(cat, cat) for cat in categories]

    # --- FÄRGER ---
    if colors is None:
        # använd vår globala färgpalett
        bar_colors = [get_category_color(col, cat) for cat in categories]
    elif isinstance(colors, dict):
        # colors kan vara t.ex. {"F": "pink", "M": "blue"}
        bar_colors = [colors.get(cat, "lightblue") for cat in categories]
    else:
        # om någon skickar in en lista med färger redan
        bar_colors = colors


    # Draw bars
    ax.bar(positions, prop.values, color=bar_colors, edgecolor="black", alpha=0.9)

    # Add text labels above bars
    for x, val in zip(positions, prop.values):
        ax.text(x, val + 1, f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")

    # Axis labels and title
    ax.set_xticks(positions)
    ax.set_xticklabels(xlabels, fontsize=11)
    ax.set_ylabel("Percentage (%)")
    ax.set_title(title if title else col, fontsize=14, pad=10)
    ax.set_ylim(0, max(prop.values) * 1.15)
    ax.grid(axis="y", alpha=0.2)

    # Show % above bars
    # for pos, v in zip(positions, prop.values):
    #     ax.text(pos, v + 1, f"{v:.1f}%", ha="center", fontsize=10)





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
    Använder plot_proportion_bar + CATEGORY_COLORS automatiskt.
    """

    if cols is None:
        cols = CAT_COLS

    pretty_titles = {
        "sex": "Sex distribution (%)",
        "smoker": "Smoking status (%)",
        "disease": "Disease status (%)",
    }

    n_cols = 2
    n_rows = int(np.ceil(len(cols) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axes = axes.flatten()

    for ax, col in zip(axes, cols):
        unique_vals = df[col].dropna().unique()

        plot_proportion_bar(
            df=df,
            col=col,
            ax=ax,
            title=pretty_titles.get(col, f"{col} (%)"),
            colors=None,            # om inget skickas in används get_category_color
            order=unique_vals,
        )

    # ta bort tomma rutor om antal inte matchar grid
    for i in range(len(cols), len(axes)):
        fig.delaxes(axes[i])

    fig.tight_layout()
    plt.show()


def plot_scatter_with_corr(
    ax,
    x,
    y,
    x_label: str,
    y_label: str,
    hue: pd.Series | None = None,
    title: str | None = None,
):
    """
    Allmän scatterplot med ev. färgkodning (hue) och korrelationskoefficient i hörnet.
    """
    data = pd.DataFrame({"x": x, "y": y})
    if hue is not None:
        data["hue"] = hue

    if hue is None:
        ax.scatter(data["x"], data["y"], alpha=0.6, edgecolor="black")
    else:
        groups = data["hue"].unique()
        for g in groups:
            subset = data[data["hue"] == g]
            color = get_category_color(hue.name if hasattr(hue, "name") else "hue", g, default=None)
            ax.scatter(
                subset["x"],
                subset["y"],
                alpha=0.6,
                edgecolor="black",
                label=f"{hue.name}: {g}",
                color=color,
            )
        ax.legend()

    # korrelation
    corr = data[["x", "y"]].corr().iloc[0, 1]
    ax.text(
        0.05,
        0.95,
        f"r = {corr:.2f}",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.6, edgecolor="gray"),
    )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.3)


def plot_correlation_heatmap(df, cols=None, title="Correlation Heatmap"):
    """
    Plots a correlation heatmap for the given columns.
    If cols is None, use NUM_COLS from visualization.
    """
    if cols is None:
        cols = NUM_COLS
    
    corr = df[cols].corr()

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        vmin=-1, vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.7}
    )
    plt.title(title)
    plt.tight_layout()
    plt.show()


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
    # --- Skapa DataFrame ---
    data = pd.DataFrame({"x": x, "y": y})

    # --- Scatterplot ---
    ax.scatter(data["x"], data["y"], alpha=0.6, edgecolor="black")

    # --- Enkel linjär regression ---
    model = LinearRegression().fit(data[["x"]], data["y"])
    
    x_vals = np.linspace(data["x"].min(), data["x"].max(), 200).reshape(-1, 1)

    # --- Använd DataFrame så feature names matchar ---
    x_vals_df = pd.DataFrame(x_vals, columns=["x"])
    y_vals = model.predict(x_vals_df)

    ax.plot(x_vals, y_vals, color="red", linewidth=2)

    # --- R² ---
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

    # --- Etiketter och titel ---
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.3)
