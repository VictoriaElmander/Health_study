import numpy as np
import pandas as pd

from power_analysis import (
    simulate_power_welch,
    required_effect_for_power,
    required_n_for_power,
    power_curve_by_effect,
    power_curve_by_n,
    summarize_power_effects,
    summarize_power_sample_size,
)
from power_plots import plot_power_vs_effect, plot_power_vs_n


class PowerAnalyzer:
    """
    Klass för power-analys av skillnad i medelvärde mellan två grupper.

    Använder:
      - simulate_power_welch  (simulerad power)
      - power_curve_by_effect (power som funktion av effektstorlek)
      - power_curve_by_n      (power som funktion av stickprovsstorlek)
      - required_effect_for_power / required_n_for_power (analytiska beräkningar)
    """

    def __init__(
        self,
        group1,
        group2,
        name1="Grupp 1",
        name2="Grupp 2",
        alpha=0.05,
        target_power=0.80,
        trials_effect=800,
        trials_n=800,
        one_sided=True,
    ):
        # spara data som rena numpy-arrayer utan NaN
        x1 = np.asarray(group1, float)
        x2 = np.asarray(group2, float)
        self.group1 = x1[~np.isnan(x1)]
        self.group2 = x2[~np.isnan(x2)]

        self.name1 = name1
        self.name2 = name2

        self.alpha = alpha
        self.target_power = target_power
        self.trials_effect = trials_effect
        self.trials_n = trials_n
        self.one_sided = one_sided

        # grundläggande storheter
        self.obs_diff = float(self.group1.mean() - self.group2.mean())
        self.real_total_n = len(self.group1) + len(self.group2)

        # cache för resultat
        self._real_power = None
        self._power_effect = None   # (effects, power_values)
        self._power_n = None        # (total_ns, power_values, n1_list, n2_list)

    # ---------- 1. Power vid observerad effekt ----------
    def real_power(self):
        """Beräknar (och cachar) power för observerad effekt Δ_obs."""
        if self._real_power is None:
            self._real_power = simulate_power_welch(
                self.group1,
                self.group2,
                effect=self.obs_diff,
                n1=len(self.group1),
                n2=len(self.group2),
                trials=self.trials_effect,
                alpha=self.alpha,
                one_sided=self.one_sided,
            )
        return self._real_power

    # ---------- 2. Power som funktion av effektstorlek ----------
    def power_by_effect(self, effects=None):
        """
        Simulerar power för en rad effektstorlekar Δ.
        Returnerar (effects, power_values).
        """
        if effects is None:
            effects = np.arange(0.0, 3.1, 0.5)

        if self._power_effect is None:
            eff, pwr = power_curve_by_effect(
                self.group1,
                self.group2,
                effects=effects,
                trials=self.trials_effect,
                alpha=self.alpha,
                one_sided=self.one_sided,
            )
            self._power_effect = (eff, pwr)

        return self._power_effect

    def plot_power_by_effect(self, effects=None):
        """Ritar power–kurvan som funktion av Δ."""
        effects, power_vals = self.power_by_effect(effects)
        plot_power_vs_effect(
            effects,
            power_vals,
            obs_diff=self.obs_diff,
            real_power=self.real_power(),
            target_power=self.target_power,
            alpha=self.alpha,
            total_n=self.real_total_n,
            group_label=f"{self.name1} − {self.name2}",
        )

        msg, _ = summarize_power_effects(
            effects, power_vals, target_power=self.target_power
        )
        return msg

    # ---------- 3. Power som funktion av stickprovsstorlek ----------
    def power_by_n(self, total_ns=None):
        """
        Simulerar power för olika totala stickprovsstorlekar.
        Returnerar (total_ns, power_values, n1_list, n2_list).
        """
        if total_ns is None:
            total_ns = np.arange(200, 1601, 200)

        if self._power_n is None:
            total_ns, pwr, n1_list, n2_list = power_curve_by_n(
                self.group1,
                self.group2,
                effect=self.obs_diff,
                total_ns=total_ns,
                trials=self.trials_n,
                alpha=self.alpha,
                one_sided=self.one_sided,
            )
            self._power_n = (total_ns, pwr, n1_list, n2_list)

        return self._power_n

    def plot_power_by_n(self, total_ns=None):
        """Ritar power–kurvan som funktion av total N."""
        total_ns, power_vals, _, _ = self.power_by_n(total_ns)

        plot_power_vs_n(
            total_ns,
            power_vals,
            real_total_n=self.real_total_n,
            real_power=self.real_power(),
            target_power=self.target_power,
            effect_chosen=self.obs_diff,
            alpha=self.alpha,
        )

        msg, _ = summarize_power_sample_size(
            total_ns, power_vals, target_power=self.target_power
        )
        return msg

    # ---------- 4. Analytiska beräkningar ----------
    def analytic_requirements(self):
        """
        Gör de analytiska beräkningarna:
        - vilken effekt som krävs med nuvarande N
        - vilket N som krävs för observerad effekt
        Returnerar en liten dict eller DataFrame.
        """
        delta_min = required_effect_for_power(
            self.group1,
            self.group2,
            total_n=None,
            power=self.target_power,
            alpha=self.alpha,
            one_sided=self.one_sided,
        )
        prop_delta = delta_min / self.obs_diff

        N_req, n1_req, n2_req = required_n_for_power(
            self.group1,
            self.group2,
            delta=self.obs_diff,
            power=self.target_power,
            alpha=self.alpha,
            one_sided=self.one_sided,
        )
        prop_N = N_req / self.real_total_n

        rows = [
            {"Typ": "Effekt givet N",
            "Delta": delta_min, 
            "Relativ storlek": prop_delta, 
            "N_totalt": self.real_total_n},
            {"Typ": "N givet effekt", 
            "Delta": self.obs_diff, 
            "Relativ storlek": prop_N, 
            "N_totalt": N_req}
        ]

        df = pd.DataFrame(rows)

        # önskad kolumnordning:
        df = df[["Typ", "Delta", "N_totalt", "Relativ storlek"]]

        # rows = [
        #     {
        #         "Typ": "Effekt givet N",
        #         "Delta": delta_min,
        #         "Relativ storlek": prop_delta,
        #         "N_totalt": self.real_total_n,
        #         "N1": len(self.group1),
        #         "N2": len(self.group2),
        #     },
        #     {
        #         "Typ": "N givet effekt",
        #         "Delta": self.obs_diff,
        #         "Relativ storlek": prop_N,
        #         "N_totalt": N_req,
        #         "N1": n1_req,
        #         "N2": n2_req,
        #     },
        # ]
        return df
