import numpy as np
import pandas as pd

from hypothesis_tests import bootstrap_mean_diff, welch_t_test_mean_diff
from hypothesis_plots import plot_mean_diff_overview


class MeanDiffAnalyzer:
    """
    Analyze the difference in means between two groups using both 
    bootstrap inference and Welch's t-test, with optional visualization.

    Intended for comparing blood pressure or other continuous measurements.
    """

    def __init__(
        self,
        group1,
        group2,
        name1="Group 1",
        name2="Group 2",
        confidence=0.95,
        two_sided=False,
        n_boot=10_000,
        random_state=42,
    ):
        # Sparar rådata (som Series/array)
        self.group1 = np.asarray(group1, dtype=float)
        self.group2 = np.asarray(group2, dtype=float)
        self.name1 = name1
        self.name2 = name2

        self.confidence = confidence
        self.two_sided = two_sided
        self.n_boot = n_boot
        self.random_state = random_state

        # Här lagras resultat efter att testerna körts
        self.bootstrap_result = None
        self.welch_result = None

    # -----------------------------
    # 1. Kör bootstrap-testet
    # -----------------------------
    def run_bootstrap(self):
        """
        Run bootstrap estimation of mean difference.

        Returns
        -------
        dict with keys:
            diff : float
                Observed difference in means.
            p_value : float
                Hypothesis test p-value from bootstrap resampling.
            ci : (low, high)
                Bootstrap confidence interval for mean difference.
            boot_diffs : np.ndarray
                Distribution of bootstrap mean differences.
        """
        
        if self.bootstrap_result is not None:
            return self.bootstrap_result
        
        diff, p_value, ci, boot_diffs = bootstrap_mean_diff(
            self.group1,
            self.group2,
            confidence=self.confidence,
            n_boot=self.n_boot,
            two_sided=self.two_sided,
            random_state=self.random_state,
        )
        self.bootstrap_result = {
            "diff": diff,
            "p_value": p_value,
            "ci": ci,
            "boot_diffs": boot_diffs,
        }
        return self.bootstrap_result

    # -----------------------------
    # 2. Kör Welch t-test
    # -----------------------------
    def run_welch(self):
        """
        Run Welch's unequal variance t-test on the two groups.

        Returns
        -------
        dict with keys:
            diff : float
                Sample mean difference.
            p_value : float
                Welch test p-value.
            ci : (low, high)
                Confidence interval for mean difference.
            t_stat : float
                Calculated t-statistic value.
            dof : float
                Degrees of freedom.
            se : float
                Standard error estimate.
        """
        if self.welch_result is not None:
            return self.welch_result
    
        diff, p_value, ci, t_stat, dof, se = welch_t_test_mean_diff(
            self.group1,
            self.group2,
            confidence=self.confidence,
            two_sided=self.two_sided,
        )
        self.welch_result = {
            "diff": diff,
            "p_value": p_value,
            "ci": ci,
            "t_stat": t_stat,
            "dof": dof,
            "se": se,
        }
        return self.welch_result
    

    # -----------------------------
    # Hjälpfunktion: etikett för grafer
    # -----------------------------
    def xlabel(self):
        """
        Return descriptive X-axis label for plots based on group names.
        """
        
        return f"Skillnad i {self.name1} - {self.name2} (mmHg)"


    # -----------------------------
    # 3. Kör båda och ge tabell
    # -----------------------------
    def summary_table(self) -> pd.DataFrame:
        """
        Run both methods and return a formatted comparison table.

        Returns
        -------
        pd.DataFrame
            Columns include:
            ['Method', 'Observed diff', 'p-value', 'CI lower', 'CI upper']
        """
        if self.bootstrap_result is None:
            self.run_bootstrap()
        if self.welch_result is None:
            self.run_welch()

        diff_b = self.bootstrap_result["diff"]
        p_b = self.bootstrap_result["p_value"]
        ci_lo_b, ci_hi_b = self.bootstrap_result["ci"]

        diff_t = self.welch_result["diff"]
        p_t = self.welch_result["p_value"]
        ci_lo_t, ci_hi_t = self.welch_result["ci"]

        rows = [
            {
                "Metod": "Bootstrap",
                "Observerad skillnad": diff_b,
                "p-värde": p_b,
                "KI nedre": ci_lo_b,
                "KI övre": ci_hi_b,
            },
            {
                "Metod": "Welch t-test",
                "Observerad skillnad": diff_t,
                "p-värde": p_t,
                "KI nedre": ci_lo_t,
                "KI övre": ci_hi_t,
            },
        ]
        df = pd.DataFrame(rows)
        return df

    # -----------------------------
    # 4. Plotta översiktsfiguren
    # -----------------------------
    def plot_overview(self):
        """
        Plot a three-panel visualization comparing:
        * Bootstrap distribution + CI
        * Welch confidence interval
        * Normal approximation overlay

        Returns
        -------
        matplotlib.Figure
        """
        if self.bootstrap_result is None:
            self.run_bootstrap()
        if self.welch_result is None:
            self.run_welch()

        diff_b = self.bootstrap_result["diff"]
        ci_lo_b, ci_hi_b = self.bootstrap_result["ci"]
        boot_diffs = self.bootstrap_result["boot_diffs"]

        diff_t = self.welch_result["diff"]
        ci_lo_t, ci_hi_t = self.welch_result["ci"]
        t_stat = self.welch_result["t_stat"]
        dof = self.welch_result["dof"]
        se = self.welch_result["se"]

        return plot_mean_diff_overview(
            boot_diffs=boot_diffs,
            obs_diff_boot=diff_b,
            ci_boot=(ci_lo_b, ci_hi_b),
            diff_t=diff_t,
            ci_welch=(ci_lo_t, ci_hi_t),
            t_stat=t_stat,
            dof=dof,
            se=se,
            confidence=self.confidence,
            xlabel=self.xlabel(),
        )
