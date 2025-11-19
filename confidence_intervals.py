import numpy as np
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