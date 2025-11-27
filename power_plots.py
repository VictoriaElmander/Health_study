import numpy as np
import matplotlib.pyplot as plt


def plot_power_vs_effect(
    effects,
    power_values,
    obs_diff=None,
    real_power=None,
    target_power=0.80,
    alpha=0.05,
    total_n=None,
    xlabel="Effektskillnad Δ (mmHg)",
    group_label="Rökare − Icke-rökare",
):
    """
    Plot statistical power as a function of effect size (Δ).

    Shows the simulated power curve, a horizontal target power line, and
    optionally marks the observed effect and its corresponding power.
    """
    plt.figure(figsize=(6, 4))
    plt.plot(effects, power_values, marker="o", label="Simulerad power (Welch)")
    plt.axhline(target_power, linestyle="--", label=f"{int(target_power*100)} % power")

    if obs_diff is not None and real_power is not None:
        power_at_obs = np.interp(obs_diff, effects, power_values)
        plt.scatter(obs_diff, power_at_obs, color="red", s=70, zorder=5)
        plt.scatter([], [], color="red", s=70,
                    label=f"Observerad power ≈ {real_power:.2f}")

    title = f"Power vs effektstorlek (α = {alpha:.2f})"
    if total_n is not None:
        title = f"Power vs effektstorlek (N = {total_n}, α = {alpha:.2f})"

    plt.title(title)
    plt.xlabel(f"{xlabel} för {group_label}")
    plt.ylabel("Power")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_power_vs_n(
    total_ns,
    power_values,
    real_total_n=None,
    real_power=None,
    target_power=0.80,
    effect_chosen=None,
    alpha=0.05,
):
    """
    Plot statistical power as a function of total sample size (N).

    Shows the simulated power curve, a horizontal target power line, and
    optionally marks the current study’s N and power for comparison.
    """
    plt.figure(figsize=(6, 4))
    plt.plot(total_ns, power_values, "o-", label="Simulerad power (Welch)")
    plt.axhline(target_power, linestyle="--", label=f"{int(target_power*100)} % power")

    if real_total_n is not None and real_power is not None:
        plt.scatter(real_total_n, real_power, color="red", s=70, zorder=5)
        plt.scatter([], [], color="red", s=70,
                    label=f"Observerad power ≈ {real_power:.2f}")

    if effect_chosen is not None:
        plt.title(
            f"Power vs total N (Δ = {effect_chosen:.2f} mmHg, α = {alpha:.2f})"
        )
    else:
        plt.title(f"Power vs total N (α = {alpha:.2f})")

    plt.xlabel("Totalt antal observationer")
    plt.ylabel("Power")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()
