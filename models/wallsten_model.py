import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import r2_score
from tqdm import tqdm


class Model3_Wallsten:
    """
    Model 3 (Wallsten et al., 2005) — Prior evaluation + Prospect theory.
    Reparametrized with q1 = a0/m0 to ensure identifiability.
    params = [gamma_plus, beta, q1, m0]
    """

    def __init__(self, reward_per_pump=5, max_pumps=128, eps=1e-12):
        self.x = float(reward_per_pump)
        self.max_pumps = int(max_pumps)
        self.eps = float(eps)

    @staticmethod
    def _g_h_stationary(gamma_plus, q_h, eps=1e-12):
        q = np.clip(q_h, eps, 1 - eps)
        return -gamma_plus / np.log(q)

    @staticmethod
    def _r_hi(beta, g_h, i):
        return expit(beta * (g_h - i))

    def _update_q(self, q, m, pumps, popped):
        """Bayesian update in q_h space"""
        if popped:
            a_h = max(0, pumps - 1)
        else:
            a_h = pumps
        m_h = pumps
        a_new = q * m + a_h
        m_new = m + m_h
        q_new = a_new / m_new
        return q_new, m_new

    def _balloon_nll_for_params(self, gamma_plus, beta, q1, m0, seq):
        if not (0 < gamma_plus <= 5 and 0 < beta <= 10 and 0 < q1 < 1 and m0 > 1):
            return 1e9

        q, m = float(q1), float(m0)
        nll = 0.0
        eps = self.eps

        for row in seq:
            pumps = int(row["pumps"])
            popped = bool(row["popped"])

            g_h = self._g_h_stationary(gamma_plus, q)

            if pumps > 0:
                i_vals = np.arange(1, pumps + 1)
                r_vals = expit(beta * (g_h - i_vals))
                nll -= np.sum(np.log(r_vals + eps))

            if not popped and pumps < self.max_pumps:
                r_stop = expit(beta * (g_h - (pumps + 1)))
                nll -= np.log(1 - r_stop + eps)

            q, m = self._update_q(q, m, pumps, popped)

        return nll

    def fit(self, user_data, n_starts=40, verbose=False):
        seq = user_data.sort_values("trial_number")[["pumps", "popped"]].to_dict("records")

        def func(p):
            return self._balloon_nll_for_params(p[0], p[1], p[2], p[3], seq)

        bounds = [(1e-3, 5.0), (0.2, 10.0), (0.01, 0.999), (2.0, 1000.0)]
        rng = np.random.default_rng(42)

        best_fun = np.inf
        best_res = None

        # reasonable start
        x0 = [0.3, 1.0, 0.95, 50.0]
        try:
            res = minimize(func, x0, method="L-BFGS-B", bounds=bounds)
            if res.success and res.fun < best_fun:
                best_fun, best_res = res.fun, res
        except Exception:
            pass

        for _ in range(n_starts):
            start = [rng.uniform(*b) for b in bounds]
            try:
                r = minimize(func, start, method="L-BFGS-B", bounds=bounds)
                if r.success and r.fun < best_fun:
                    best_fun, best_res = r.fun, r
            except Exception:
                continue

        if verbose and best_res is not None:
            print(f"fit done. nll={best_fun:.3f}, params={best_res.x}")

        return best_res

    def simulate(self, params, n_trials, seed=None):
        gamma_plus, beta, q1, m0 = params
        rng = np.random.default_rng(seed)
        q, m = float(q1), float(m0)
        rows = []

        for h in range(1, n_trials + 1):
            g_h = self._g_h_stationary(gamma_plus, q)
            explosion_point = rng.integers(1, self.max_pumps + 1)
            j = 0
            popped = False

            while True:
                j += 1
                r = expit(beta * (g_h - j))
                if rng.random() >= r:
                    popped = False
                    pumps = j - 1
                    break
                if j >= explosion_point:
                    popped = True
                    pumps = j
                    break
                if j >= self.max_pumps:
                    popped = False
                    pumps = j
                    break

            rows.append({"trial_number": h, "pumps": pumps, "popped": popped})
            q, m = self._update_q(q, m, pumps, popped)

        return pd.DataFrame(rows)

    def predictive_check(self, real_data, seed=42, verbose=True):
        res = self.fit(real_data, verbose=verbose)
        if res is None:
            raise RuntimeError("fit failed")
        params = res.x
        sim = self.simulate(params, len(real_data), seed)
        merged = pd.merge(real_data[["trial_number", "pumps"]], sim[["trial_number", "pumps"]],
                          on="trial_number", suffixes=("_real", "_sim"))
        r2 = r2_score(merged["pumps_real"], merged["pumps_sim"])
        if verbose:
            print(f"mean real={real_data['pumps'].mean():.2f}, sim={sim['pumps'].mean():.2f}, R2={r2:.3f}")
        return {"r2": r2, "params": params, "sim": sim, "res": res}

    def parameter_recovery(self, n_subjects=20, n_trials=90, seed=123):
        rng = np.random.default_rng(seed)
        true_list, fit_list = [], []

        for i in tqdm(range(n_subjects), desc="Model3-Q recovery"):
            gamma_true = rng.uniform(0.1, 1.5)
            beta_true = rng.uniform(0.5, 3.0)
            q1_true = rng.uniform(0.9, 0.99)
            m0_true = rng.uniform(10, 200)
            p_true = [gamma_true, beta_true, q1_true, m0_true]
            sim = self.simulate(p_true, n_trials, seed=1000 + i)

            try:
                res = self.fit(sim, verbose=False)
                p_fit = res.x if res is not None else [np.nan] * 4
            except Exception:
                p_fit = [np.nan] * 4

            true_list.append(p_true)
            fit_list.append(p_fit)

        cols = ["gamma_plus", "beta", "q1", "m0"]
        df = pd.DataFrame(np.hstack([np.array(true_list), np.array(fit_list)]),
                          columns=[f"true_{c}" for c in cols] + [f"fit_{c}" for c in cols])

        print("Parameter recovery correlations:")
        for c in cols:
            r = np.corrcoef(df[f"true_{c}"], df[f"fit_{c}"])[0, 1]
            print(f" {c}: r={r:.3f}")
        return df
