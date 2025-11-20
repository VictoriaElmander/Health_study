import numpy as np
import pandas as pd
from scipy.stats import norm

def ci_prop_normal(prop: float, n: int, level: float = 0.95, clip: bool = True):
    """
    Compute a confidence interval for a proportion using the normal approximation.
    
    Parameters
    ----------
    prop : float
        Observed proportion (between 0 and 1).
    n : int
        Sample size.
    level : float
        Confidence level (default: 0.95).
    clip : bool
        Clip CI into [0, 1] since proportions cannot be outside the range.
        
    Returns
    -------
    (lo, hi, sd, z)
        Lower bound, upper bound, standard error, and z-value.
    """
    alpha = 1 - level
    z = norm.ppf(1 - alpha/2)
    sd = np.sqrt(prop * (1 - prop) / n)
    lo, hi = prop - z*sd, prop + z*sd
    
    if clip:
        lo, hi = max(0.0, lo), min(1.0, hi)
    
    return lo, hi, sd, z


# Confidence interval using normal approximation
def ci_norm(series, confidence = 0.95):
    # Validering
    if not (0 < confidence < 1):
        raise ValueError("confidence måste ligga mellan 0 och 1.")
    
    x = np.asarray(series, dtype=float)
    x = x[~np.isnan(x)]
    n=len(x)
    if n == 0:
        raise ValueError("Ingen data kvar efter dropna().")
    
    mean = x.mean()
    std = x.std(ddof=1)
    
    alpha=1-confidence
    # z-value for the chosen confidence level
    z = norm.ppf(1 - alpha/2)

    se = std/np.sqrt(n) #standard error

    ci_lower = mean - z*se
    ci_upper = mean + z*se

    level = int(round(confidence*100))

    return  float(mean), (float(ci_lower), float(ci_upper)), level  


# Confidence interval using bootstrap Percentile CI
def ci_mean_boot_perc(series, confidence=0.95, n_boot=20_000, random_state=42):
    # Validation
    if not (0 < confidence < 1):
        raise ValueError("Confidence must be between 0 and 1.")
    
    x = np.asarray(series, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        raise ValueError("No data remaining after dropna().")
    
    mean_x = float(x.mean())
    rng = np.random.default_rng(random_state)

    # Ver1 Bootstrap Means via Index Sampling
    idx = rng.integers(0, n, size=(n_boot,n))
    boot_mean = x[idx].mean(axis=1)

    # Ver 2. LOOP: Create one resample at a time
    # boot_mean = np.empty(n_boot,dtype=float)

    # for i in range(n_boot):
    #     boot_sample = rng.choice(x, size=n, replace=True)
    #     boot_mean[i] = boot_sample.mean()

    alpha = 1 - confidence
    lo = float(np.quantile(boot_mean, alpha/2)) # At alpha = 0.5, 2.5% of the bootstrap means lie below lo
    hi = float(np.quantile(boot_mean, 1-alpha/2)) #See above, the threshold below which 97.5% lie at alpha = 0.5

    level = int(round(confidence*100))

    return mean_x, (lo, hi), level #boot_mean
   

# Confidence interval using bootstrap BCa
def ci_mean_boot_bca(series, confidence=0.95, n_boot=20_000, random_state=42):
    # Validation
    if not (0 < confidence < 1):
        raise ValueError("Confidence must be between 0 and 1.")
    
    x = np.asarray(series, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        raise ValueError("No data remaining after dropna().")
    
    mean_x = float(x.mean())
    rng = np.random.default_rng(random_state)

    # Bootstrap with resampling with replacement
    idx = rng.integers(0, n, size=(n_boot,n))
    boot_mean = x[idx].mean(axis=1)

    # Bootstrap correction (z0)
    # Proportion of bootstrap means < observed mean
    prop = (boot_mean < mean_x).mean() #If not 0.5, the bootstrap distribution is shifted (bias)
    z0 = norm.ppf(prop) #Correction factor, transformed into a z-value

    # Acceleration(a) — adjusts for skewness, based on the jackknife 
    sumx = x.sum()
    loo_means = (sumx - x) / (n - 1)  # LOO-mean: the mean when each observation is left out in turn
    jack_mean = loo_means.mean() #Average of LOO-means

    num = np.sum((jack_mean - loo_means)**3)
    den = 6 * (np.sum((jack_mean - loo_means)**2))**1.5
    a = num/den if den != 0 else 0.0 # If 0, there is no skewness correction

    #BCa quantile adjustment
    alpha = 1 - confidence

    def bca_quantile(p):
        z = norm.ppf(p)
        adj = z0 + (z0 + z) / (1 - a * (z0 + z))   # BCa transformation
        return float(np.quantile(boot_mean, norm.cdf(adj)))

    lo = bca_quantile(alpha/2)
    hi = bca_quantile(1 - alpha/2)

    level = int(round(confidence*100))

    return mean_x, (lo, hi), level #boot_mean

import pandas as pd

# Skapa jämförelsetabell för CI på riktiga data
def ci_methods_table(series, confidence=0.95, n_boot=20_000, random_state=42):
    """ Compare CI results on real data """
    mean_n, (ci_lo_n, ci_hi_n), level_n = ci_norm(series, confidence)
    mean_bp, (ci_lo_bp, ci_hi_bp), level_bp = ci_mean_boot_perc(series, confidence, n_boot, random_state)
    mean_bbca, (ci_lo_bbca, ci_hi_bbca), level_bbca = ci_mean_boot_bca(series, confidence, n_boot, random_state)

    rows = [
        {"Metod": "Normalapproximation",   "Medel": mean_n,   "KI nedre": ci_lo_n,   "KI övre": ci_hi_n,   "Nivå (%)": level_n},
        {"Metod": "Bootstrap (percentil)", "Medel": mean_bp,  "KI nedre": ci_lo_bp,  "KI övre": ci_hi_bp,  "Nivå (%)": level_bp},
        {"Metod": "Bootstrap (BCa)",       "Medel": mean_bbca,"KI nedre": ci_lo_bbca,"KI övre": ci_hi_bbca,"Nivå (%)": level_bbca},
    ]
    return pd.DataFrame(rows)
