import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t as t_dist


def plot_bootstrap_dist(boot_diffs, obs_diff, ci,  p_boot=None, xlabel="Skillnad (grupp1 − grupp2)"):
    ci_low, ci_high = ci
    
    fig, ax = plt.subplots(figsize=(7,4))
    ax.hist(boot_diffs, bins=50, density=True, alpha=0.6, label="Bootstrapfördelning")
    
    # Observed difference
    ax.axvline(obs_diff, linewidth=2, label=f'Observerad diff = {obs_diff:.3f}')

    # Confidence interval
    ax.axvline(ci_low, linestyle='--', label=f"KI: ({ci_low:.3f}, {ci_high:.3f})")
    ax.axvline(ci_high, linestyle='--')
    
    # Zero difference (for intuition)
    ax.axvline(0, linestyle=':', label='0')

    ax.set_title("Bootstrapfördelning av skillnad i medelvärde")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Täthet")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


def plot_t_distribution(t_stat, dof, confidence=0.95, two_sided=False):
    """
    Plottar t-fördelningen med markerat kritiskt område och observerat t-värde.
    two_sided=False = ensidigt test (H1: diff > 0).
    """

    alpha = 1 - confidence
    
    # Determine critical t depending on one-sided/two-sided test
    if two_sided:
        t_crit = t_dist.ppf(1 - alpha/2, dof)
        crit_left = -t_crit
        crit_right = t_crit
    else:
        t_crit = t_dist.ppf(1 - alpha, dof)
        crit_left = None
        crit_right = t_crit

    # x-axis
    xs = np.linspace(-5, 5, 1000)
    ys = t_dist.pdf(xs, dof)

    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(xs, ys, label=f"t-fördelning (df = {dof:.1f})")

    # Shade the critical region
    if two_sided:
        ax.fill_between(xs, 0, ys, where=(xs <= crit_left), alpha=0.2, label="Kritiskt område (α/2)")
        ax.fill_between(xs, 0, ys, where=(xs >= crit_right), alpha=0.2)
    else:
        ax.fill_between(xs, 0, ys, where=(xs >= crit_right), alpha=0.2, label="Kritiskt område (α)")
    
    # Mark the critical t-value
    if two_sided:
        ax.axvline(crit_left, linestyle='--', label=f"−t_crit = {crit_left:.3f}")
    ax.axvline(crit_right, linestyle='--', label=f"t_crit = {t_crit:.3f}")

    # Mark the observed t-value
    ax.axvline(t_stat, linewidth=2, label=f"t_obs = {t_stat:.3f}", color="black")

    # Graph settings
    title = "t-fördelning med kritiskt område (ensidigt test)" if not two_sided else "t-fördelning med kritiskt område (tvåsidigt test)"
    ax.set_title(title)
    ax.set_xlabel("t-värde")
    ax.set_ylabel("Täthet")
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


def plot_mean_diff_normal(diff, se, ci, xlabel="Skillnad (grupp1 − grupp2)"):
    """
    Visualizes the normal approximation of the difference in means.
    'diff' = observed difference.
    'se'   = standard error of the difference.
    'ci'   = (ci_low, ci_high) from your test.
    """
    
    ci_low, ci_high = ci

    # Create x-axis around diff
    xs = np.linspace(diff - 5*se, diff + 5*se, 400)
    ys = (1/(np.sqrt(2*np.pi)*se)) * np.exp(-(xs - diff)**2 / (2*se**2))

    plt.figure(figsize=(7,4))
    plt.plot(xs, ys, label="Normalapproximation")
    
    # Observed difference
    plt.axvline(diff, linewidth=2, label=f"Observerad diff = {diff:.3f}")

    # Confidence interval
    plt.axvline(ci_low, linestyle='--', label=f"KI: ({ci_low:.3f}, {ci_high:.3f})")
    plt.axvline(ci_high, linestyle='--')
    plt.fill_between(xs, 0, ys, where=(xs>=ci_low)&(xs<=ci_high), alpha=0.2)

    # Zero difference
    plt.axvline(0, linestyle=':', label="0")

    plt.title("Normalapproximerad fördelning av skillnad i medelvärde")
    plt.xlabel(xlabel)
    plt.ylabel("Täthet")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    
def plot_mean_diff_overview(
    boot_diffs,
    obs_diff_boot,
    ci_boot,
    diff_t,
    ci_welch,
    t_stat,
    dof,
    se,
    confidence=0.95,
    xlabel="Skillnad i systoliskt blodtryck (mmHg)"):
    """
    Ritar en 3-delad figur:
      1) t-fördelning med kritiskt område och observerat t-värde
      2) Normalapproximerad fördelning för skillnaden i medelvärde (Welch)
      3) Bootstrapfördelning för skillnaden i medelvärde

    Parameters
    ----------
    boot_diffs : array-like
        Bootstrapfördelning av skillnad i medelvärde.
    obs_diff_boot : float
        Observerad skillnad i medelvärde (från bootstrap-anropet, samma som diff_t normalt).
    ci_boot : (lo, hi)
        KI från bootstrap-metoden.
    diff_t : float
        Observerad skillnad i medelvärde från Welch t-test.
    ci_welch : (lo, hi)
        KI från Welch t-test.
    t_stat : float
        t-observerat från Welch t-test.
    dof : float
        Frihetsgrader från Welch t-test.
    se : float
        Standard error för skillnaden (från Welch).
    xlabel : str
        Etikett för x-axlar (skillnad i medel).
    """

    ci_low_wt, ci_high_wt = ci_welch
    ci_low_b, ci_high_b = ci_boot

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # --- 1️⃣ t-fördelning ---
    alpha = 1-confidence
    t_crit = t_dist.ppf(1 - alpha, dof)
    xs = np.linspace(-5, 5, 400)
    ys = t_dist.pdf(xs, dof)

    axes[0].plot(xs, ys, label=f"t-fördelning (df = {dof:.1f})")
    axes[0].fill_between(xs, 0, ys, where=(xs >= t_crit), alpha=0.2, label="Kritiskt område (α)")
    axes[0].axvline(t_crit, ls="--", label=f"t_crit = {t_crit:.3f}")
    axes[0].axvline(t_stat, lw=2, label=f"t_obs = {t_stat:.3f}")
    axes[0].axvline(0, ls=":", label="0")
    axes[0].set_title("t-fördelning med kritiskt område")
    axes[0].set_xlabel("t-värde")
    axes[0].set_ylabel("Täthet")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=8)

    # --- 2️⃣ Normalapproximation (Welch) ---
    xs = np.linspace(diff_t - 5 * se, diff_t + 5 * se, 400)
    ys = (1 / (np.sqrt(2 * np.pi) * se)) * np.exp(-(xs - diff_t) ** 2 / (2 * se**2))

    axes[1].plot(xs, ys, label="Normalapproximation")
    axes[1].axvline(diff_t, lw=2, label=f"Observerad diff = {diff_t:.3f}")
    axes[1].axvline(ci_low_wt, ls="--")
    axes[1].axvline(ci_high_wt, ls="--", label=f"KI: [{ci_low_wt:.3f}, {ci_high_wt:.3f}]")
    axes[1].axvline(0, ls=":", label="0")
    axes[1].fill_between(
        xs, 0, ys, where=(xs >= ci_low_wt) & (xs <= ci_high_wt), alpha=0.2
    )
    axes[1].set_title("Normalapproximerad fördelning av skillnad i medelvärde")
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("Täthet")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper left", fontsize=8)

    # --- 3️⃣ Bootstrapfördelning ---
    axes[2].hist(boot_diffs, bins=50, density=True, alpha=0.6, label="Bootstrapfördelning")
    axes[2].axvline(obs_diff_boot, lw=2, label=f"Observerad diff = {obs_diff_boot:.3f}")
    axes[2].axvline(ci_low_b, ls="--")
    axes[2].axvline(ci_high_b, ls="--", label=f"KI: [{ci_low_b:.3f}, {ci_high_b:.3f}]")
    axes[2].axvline(0, ls=":", label="0")
    axes[2].set_title("Bootstrapfördelning av skillnad i medelvärde")
    axes[2].set_xlabel(xlabel)
    axes[2].set_ylabel("Täthet")
    axes[2].grid(alpha=0.3)
    axes[2].legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.show()

    return fig, axes
