import numpy as np
from scipy.stats import ttest_ind, norm

# ==============================================
# Simulated Power Function
# ==============================================
def simulate_power_welch(
    group1, group2,
    effect=1.0,           # true difference in means: group1 - group2 (mmHg)
    n1=None, n2=None,     # sample sizes; None = keep current sizes
    trials=1000,
    alpha=0.05,
    one_sided=True,
    random_state=42
    ):
    """
    Power = the proportion of simulations where p < alpha given the true effect ‘effect’.
    Preserves shape/variance by resampling the centered data and shifting the mean in group1 by ‘effect’.
    Test: Welch’s t-test (equal_var=False).
    """
    rng = np.random.default_rng(random_state)

    # Convert to clean NumPy arrays (remove NaNs)
    x1 = np.asarray(group1, dtype=float)
    x2 = np.asarray(group2, dtype=float)
    x1 = x1[~np.isnan(x1)]
    x2 = x2[~np.isnan(x2)]

    # center to preserve shape/variance
    x1c = x1 - x1.mean()
    x2c = x2 - x2.mean()

    if n1 is None: n1 = len(x1) # Allows other sizes, default = current lengths
    if n2 is None: n2 = len(x2)

    hits = 0
    for _ in range(trials):
        # resample and shift group1 by ‘effect’
        s1 = rng.choice(x1c, size=n1, replace=True) + x1.mean() + effect
        s2 = rng.choice(x2c, size=n2, replace=True) + x2.mean()

        t_stat, p_two = ttest_ind(s1, s2, equal_var=False)

        if one_sided:
            # H1: mean(group1) > mean(group2)  (adjust the sign if needed)
            p = p_two/2 if t_stat > 0 else 1 - p_two/2
        else:
            p = p_two

        hits += (p < alpha)

    return hits / trials

# =====================================================================
# Functions for calculating what is required to achieve Power = 80% 
# Analytical calculation, not simulation, assuming normal approximation
# =====================================================================

# minimum Δ for a given N (and power)
def required_effect_for_power(group1, group2, 
                              total_n=None, power=0.8,
                              alpha=0.05, one_sided=True):
    """
    Minimum detectable effect Δ for target power.
    Assumes normal approx and preserved group ratio.
    """
    x1 = np.asarray(group1, float); x1 = x1[~np.isnan(x1)]
    x2 = np.asarray(group2, float); x2 = x2[~np.isnan(x2)]
    
    # Pooled SD (shared sigma)
    all_x = np.concatenate([x1, x2])
    sigma = all_x.std(ddof=1)

    # Group proportions (use the current distribution)
    n1 = len(x1); n2 = len(x2)
    if total_n is None:
        total_n = n1 + n2
    p = n1 / (n1 + n2)   # proportion of smokers (group 1)
    
    C = 1/p + 1/(1-p)    # factor from 1/(pN) + 1/((1–p)N)

    # Critical z-values
    if one_sided:
        z_crit = norm.ppf(1 - alpha)
    else:
        z_crit = norm.ppf(1 - alpha/2)
    z_power = norm.ppf(power)  # e.g., ≈ 0.84 at power 0.80

    # Minimum Δ:
    delta_min = sigma * np.sqrt(C / total_n) * (z_crit + z_power)
    return float(delta_min)

def required_n_for_power(group1, group2,
                         delta, power=0.8,
                         alpha=0.05, one_sided=True):
    """
    Required total N to detect effect Δ at target power.
    Group proportions preserved.
    """
    x1 = np.asarray(group1, float); x1 = x1[~np.isnan(x1)]
    x2 = np.asarray(group2, float); x2 = x2[~np.isnan(x2)]
    
    # Pooled standard deviation
    all_x = np.concatenate([x1, x2])
    sigma = all_x.std(ddof=1)
    
    # Proportion between the groups (keep the same as in the data)
    n1_now = len(x1); n2_now = len(x2)
    p = n1_now / (n1_now + n2_now)   # proportion of smokers
    C = 1/p + 1/(1-p)
    
    # z-value
    if one_sided:
        z_crit = norm.ppf(1 - alpha)
    else:
        z_crit = norm.ppf(1 - alpha/2)
    z_power = norm.ppf(power)

    # Solve for N from the formula:
    N_total = (sigma**2 * C * (z_crit + z_power)**2) / (delta**2)
    
    # Split N_total into groups according to the same proportion
    n1 = int(round(p * N_total))
    n2 = int(round(N_total - n1))
    
    return int(round(N_total)), n1, n2


def power_curve_by_effect(
    group1,
    group2,
    effects,
    trials=800,
    alpha=0.05,
    one_sided=True,
    random_state=42,
):
    """
    Power values for a range of effect sizes.
    Returns (effects, power).
    """
    effects = np.asarray(effects, float)
    power_vals = [
        simulate_power_welch(
            group1,
            group2,
            effect=d,
            trials=trials,
            alpha=alpha,
            one_sided=one_sided,
            random_state=random_state,
        )
        for d in effects
    ]
    return effects, np.array(power_vals)


def power_curve_by_n(
    group1,
    group2,
    effect,
    total_ns,
    trials=600,
    alpha=0.05,
    one_sided=True,
    random_state=42,
):
    """
    Power as function of total N, keeping group ratio.
    Returns (total_ns, power, n1, n2).
    """
    x1 = np.asarray(group1, float); x1 = x1[~np.isnan(x1)]
    x2 = np.asarray(group2, float); x2 = x2[~np.isnan(x2)]

    prop = len(x1) / (len(x1) + len(x2))  # andel i grupp 1

    total_ns = np.asarray(total_ns, int)
    power_vals = []
    n1_list = []
    n2_list = []

    for N in total_ns:
        n1 = int(round(N * prop))
        n2 = N - n1
        n1 = max(n1, 5)
        n2 = max(n2, 5)

        pwr = simulate_power_welch(
            x1,
            x2,
            effect=effect,
            n1=n1,
            n2=n2,
            trials=trials,
            alpha=alpha,
            one_sided=one_sided,
            random_state=random_state,
        )
        power_vals.append(pwr)
        n1_list.append(n1)
        n2_list.append(n2)

    return total_ns, np.array(power_vals), np.array(n1_list), np.array(n2_list)


def summarize_power_effects(effects, power_values, target_power=0.80):
    """
    Return minimum Δ reaching target power (or None).
    """
    max_power = power_values.max()

    if max_power >= target_power:
        delta_at_target = float(np.interp(target_power, power_values, effects))
        message = (
            f"För att nå {target_power*100:.0f} % power med nuvarande stickprovsstorlek\n"
            f"krävs en sann effekt på ungefär Δ ≈ {delta_at_target:.2f} mmHg."
        )
        return message, delta_at_target
    else:
        message = (
            f"Även vid den största testade effekten (Δ = {effects.max():.2f} mmHg)\n"
            f"når power inte upp till {target_power*100:.0f} %."
        )
        return message, None
    

def summarize_power_sample_size(total_ns, power_values, target_power=0.80):
    """
    Return minimum total N reaching target power (or None).
    """
    max_power = power_values.max()

    if max_power >= target_power:
        N_at_target = float(np.interp(target_power, power_values, total_ns))
        message = (
            f"För att nå {target_power*100:.0f} % power\n"
            f"behövs ungefär N ≈ {N_at_target:.0f} observationer totalt."
        )
        return message, N_at_target
    else:
        message = (
            f"Power når inte {target_power*100:.0f} % inom intervallet "
            f"N = {total_ns.min()}–{total_ns.max()}."
        )
        return message, None