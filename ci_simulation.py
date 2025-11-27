import numpy as np
import pandas as pd
from confidence_intervals import ci_norm, ci_mean_boot_perc, ci_mean_boot_bca


def covers_true_mean(series, method="normal", n=40, trials=200, confidence=0.95, n_boot=20_000, random_state=42):
    """
    Calculates the coverage for a given CI method.

    Args:
        series: Data to analyze (pd.Series or array)
        method: Which CI method to use ('normal', 'bootstrap', or 'bca')
        n: Sample size for each simulation
        trials: Number of simulations to run
        confidence: Confidence level (e.g., 0.95 for 95% CI)
        random_state: Random seed for reproducibility

    Returns:
        Dict with the method's performance:
        - coverage: Proportion of intervals that contain the population mean
        - avg_width: Average interval width
        - std_width: Standard deviation of the interval widths
    """
    x = np.asarray(series, dtype=float)
    x = x[~np.isnan(x)]
    true_mean = float(x.mean())
    rng = np.random.default_rng(random_state)

    hits = 0 # Counter for number of hits (intervals that cover true_mean)
    widths = [] # Collecting interval widths

    # Select CI method
    func_map = {
        "normal": ci_norm,
        "bootstrap": ci_mean_boot_perc,
        "bca": ci_mean_boot_bca
    }
    ci_func = func_map.get(method)
    if ci_func is None:
        raise ValueError(f"Unknown method: {method}")
    
    # Run simulations
    for _ in range(trials):
        sample = rng.choice(x, size=n, replace=True)
        
        # Calculate CI 
        if method in ["bootstrap", "bca"]:
            out = ci_func(sample, confidence, n_boot=n_boot) #, random_state=random_state så inte exakt samma slumpsekvens används för bootstrap-resamplingen.
        else:
            out = ci_func(sample, confidence)

        # Unpack the result
        _, (lo, hi), _ = out
        
        #Update statistics
        hits += (lo <= true_mean <= hi)
        widths.append(hi - lo)
    
    return {
        'method': method,
        'coverage': hits / trials,
        'avg_width': float(np.mean(widths)),
        'std_width': float(np.std(widths))
    }


#Simulera och utvärdera CI-metoderna (täckningsgrad, bredd, stabilitet).
def evaluate_ci_methods(data, methods=('normal','bootstrap','bca'), 
                       n=40, trials=200, confidence=0.95,
                       n_boot=20_000, random_state=42):
    """
    Compare multiple CI methods through simulation.
    
    Args:
        data: Dataset to analyze
        methods: List of methods to compare
        n, trials, confidence: Passed on to covers_true_mean
    
    Returns:
        DataFrame with results for all methods
    """
    results = [covers_true_mean(data, 
                                method=m, 
                                n=n, 
                                trials=trials, 
                                confidence=confidence,
                                n_boot=n_boot,
                                random_state=random_state
                                ) for m in methods]
    result_df = pd.DataFrame(results).set_index('method')
    return result_df
