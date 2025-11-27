# Function definitions for testing mean differences: bootstrap and Welch’s t-test
import numpy as np
from scipy.stats import ttest_ind, t as t_dist

def bootstrap_mean_diff(group1, group2, confidence=0.95, n_boot=10_000, 
                        two_sided=False, random_state=42):
    """
    Bootstrap test for the difference in means between two groups.
    Returns the observed difference, p-value, confidence interval, and bootstrap distribution.
    """

    # Convert to clean NumPy arrays (remove NaNs)
    x1 = np.asarray(group1, dtype=float)
    x2 = np.asarray(group2, dtype=float)
    x1 = x1[~np.isnan(x1)]
    x2 = x2[~np.isnan(x2)]

    obs_diff = x1.mean() - x2.mean()

    rng = np.random.default_rng(random_state)
    boot_diffs = np.empty(n_boot)

    # Bootstrap resampling
    for i in range(n_boot):
        s1 = rng.choice(x1, size=len(x1), replace=True)
        s2 = rng.choice(x2, size=len(x2), replace=True)
        boot_diffs[i] = s1.mean() - s2.mean()

    # p-value
    if two_sided:
        p_boot = np.mean(np.abs(boot_diffs) >= np.abs(obs_diff))
    else:
        p_boot = np.mean(boot_diffs >= obs_diff)

    # Confidence interval
    alpha = 1 - confidence
    ci_low, ci_high = np.percentile(boot_diffs, [100*alpha/2, 100*(1-alpha/2)])

    return obs_diff, p_boot, (ci_low, ci_high), boot_diffs


#t-test for difference in means between two groups
def welch_t_test_mean_diff(group1, group2, confidence = 0.95, two_sided=False):
    """
    Welch's t-test for the difference in means between two groups.
    Returns:
        diff (difference in means),
        p_val (p-value),
        (ci_low, ci_high) (confidence interval),
        t_stat (t-statistic),
        dof (degrees of freedom).
    """
    # Convert to clean NumPy arrays (remove NaNs)
    x1 = np.asarray(group1, dtype=float)
    x2 = np.asarray(group2, dtype=float)
    x1 = x1[~np.isnan(x1)]
    x2 = x2[~np.isnan(x2)]

    # Means & variances
    m1, m2 = x1.mean(), x2.mean()
    v1, v2 = x1.var(ddof=1), x2.var(ddof=1)
    n1, n2 = len(x1), len(x2)

    # Observed difference in means
    diff = m1 - m2

    # Welch's t-test using SciPy
    t_stat, p_two_sided = ttest_ind(x1, x2, equal_var=False)

    #Handle one-sided vs two-sided p-value
    if two_sided:
        p_value = p_two_sided
    else:
        # One-sided test: tests the hypothesis diff > 0
        p_value = p_two_sided/2 if t_stat > 0 else 1 - p_two_sided/2
 
    # Degrees of freedom (Welch–Satterthwaite)
    dof = (v1/n1 + v2/n2)**2 / ((v1**2)/((n1**2)*(n1-1)) + (v2**2)/((n2**2)*(n2-1)))

    # Standard error and critical t-value
    se = np.sqrt(v1/n1 + v2/n2)
    alpha = 1 - confidence
    t_crit = t_dist.ppf(1 - alpha/2, dof) if two_sided else t_dist.ppf(1 - alpha, dof)

    # Confidence interval
    ci_low = diff - t_crit * se
    ci_high = diff + t_crit * se

    return diff, p_value, (ci_low, ci_high), t_stat, dof, se