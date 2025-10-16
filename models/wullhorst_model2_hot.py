import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import r2_score
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

class HOTModel2:
    """
    Wüllhorst Model 2 (Prospect Theory + Prelec probability weighting) for hot CCT.

    Parameters:
    ----------
    rho (ρ): curvature of utility (risk attitude)
    lam (λ): loss aversion
    delta (δ): Prelec elevation parameter
    eta (η): Prelec shape parameter
    beta (β): decision noise ("inverse temperature")
    """

    def __init__(self, data_df):
        self.data = data_df.copy()

    # ---------- utility & weighting ----------
    @staticmethod
    def utility(x, rho, lam):
        x = np.asarray(x, float)
        u = np.power(np.abs(x), rho)
        u[x < 0] *= -lam
        return u

    @staticmethod
    def prelec_weight(p, delta, eta, eps=1e-12):
        p = np.clip(p, eps, 1 - eps)
        return np.exp(-delta * (-np.log(p)) ** eta)

    # ---------- fast negative log-likelihood ----------
    def nll(self, params, flips_df):
        rho, lam, delta, eta, beta = params
        eps = 1e-12

        gain = flips_df["gain_amount"].to_numpy()
        loss = flips_df["loss_amount"].to_numpy()
        flip_no = flips_df["flip_number"].to_numpy()
        loss_cards = flips_df["loss_cards"].to_numpy()
        choices = flips_df["choice"].to_numpy()
        total_cards = (
            flips_df["total_cards"].to_numpy()
            if "total_cards" in flips_df.columns
            else np.full_like(flip_no, 32)
        )

        # same gain/loss per trial: take per-row via grouping
        gain_per_trial = flips_df.groupby("trial_number")["gain_amount"].transform("first").to_numpy()
        loss_per_trial = flips_df.groupby("trial_number")["loss_amount"].transform("first").to_numpy()
        total_per_trial = (
            flips_df.groupby("trial_number")["total_cards"].transform("first").to_numpy()
            if "total_cards" in flips_df.columns
            else np.full_like(flip_no, 32)
        )

        denom = np.maximum(total_per_trial - (flip_no - 1), 1)
        p_loss = loss_cards / denom
        p_gain = 1.0 - p_loss

        u_gain = self.utility(gain_per_trial, rho, lam)
        u_loss = self.utility(-np.abs(loss_per_trial), rho, lam)
        w_gain = self.prelec_weight(p_gain, delta, eta)
        w_loss = self.prelec_weight(p_loss, delta, eta)

        EU = w_gain * u_gain + w_loss * u_loss
        p_turn = expit(EU / beta)
        p_turn = np.clip(p_turn, eps, 1 - eps)

        return -np.sum(choices * np.log(p_turn) + (1 - choices) * np.log(1 - p_turn))

    # ---------- parallelized multi-start fit ----------
    def fit(self, flips_df, bounds=None, n_starts=50, n_jobs=8, random_state=None):
        if bounds is None:
            bounds = [
                (0.01, 3.0), (0.01, 10.0),
                (0.01, 5.0), (0.01, 3.0),
                (0.01, 1.0)
            ]
        rng = np.random.default_rng(random_state)

        def one_start(_):
            x0 = [rng.uniform(b[0], b[1]) for b in bounds]
            try:
                res = minimize(self.nll, x0, args=(flips_df,), bounds=bounds, method="L-BFGS-B")
                if res.success:
                    return res.fun, res.x
            except Exception:
                return np.inf, None
            return np.inf, None

        best_fun, best_x = np.inf, None
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            futures = [ex.submit(one_start, i) for i in range(n_starts)]
            for f in tqdm(as_completed(futures), total=n_starts, desc="Multistart fit", leave=False):
                fun, x = f.result()
                if fun < best_fun:
                    best_fun, best_x = fun, x

        return best_x if best_x is not None else [np.nan] * 5

    def fit_with_choice(self, flips_df, **kw):
        return self.fit(flips_df, **kw)

    # ---------- simulate ----------
    def simulate(self, params, flips_df, seed=None):
        rho, lam, delta, eta, beta = params
        rng = np.random.default_rng(seed)
        sim = flips_df.drop(columns=["choice"], errors="ignore").copy()

        gain = sim.groupby("trial_number")["gain_amount"].transform("first").to_numpy()
        loss = sim.groupby("trial_number")["loss_amount"].transform("first").to_numpy()
        total = (
            sim.groupby("trial_number")["total_cards"].transform("first").to_numpy()
            if "total_cards" in sim.columns
            else np.full_like(sim["flip_number"].to_numpy(), 32)
        )
        denom = np.maximum(total - (sim["flip_number"].to_numpy() - 1), 1)
        p_loss = sim["loss_cards"].to_numpy() / denom
        p_gain = 1 - p_loss

        u_gain = self.utility(gain, rho, lam)
        u_loss = self.utility(-np.abs(loss), rho, lam)
        w_gain = self.prelec_weight(p_gain, delta, eta)
        w_loss = self.prelec_weight(p_loss, delta, eta)
        EU = w_gain * u_gain + w_loss * u_loss
        p_turn = expit(EU / beta)
        sim["choice"] = rng.binomial(1, np.clip(p_turn, 1e-12, 1 - 1e-12))
        return sim

    # ---------- predictive check ----------
    def predictive_check(self, user_flips, fit_kwargs=None, seed=42):
        fit_kwargs = fit_kwargs or {}
        params = self.fit(user_flips, **fit_kwargs)
        sim = self.simulate(params, user_flips, seed)
        r2 = r2_score(user_flips["choice"], sim["choice"])
        print(f"PPC: ρ={params[0]:.3f}, λ={params[1]:.3f}, δ={params[2]:.3f}, η={params[3]:.3f}, β={params[4]:.3f}, R²={r2:.3f}")
        return r2

    # ---------- parameter recovery ----------
    def parameter_recovery(self, template_flips, n_subjects=20, n_starts=30, n_jobs=8, random_state=123):
        rng = np.random.default_rng(random_state)
        bounds = [(0.01,3.0),(0.01,10.0),(0.01,5.0),(0.01,3.0),(0.01,1.0)]
        true_params = np.column_stack([rng.uniform(b[0], b[1], n_subjects) for b in bounds])

        def one(i):
            tp = true_params[i]
            sim = self.simulate(tp, template_flips, seed=100+i)
            fit = self.fit(sim, bounds=bounds, n_starts=n_starts, n_jobs=4, random_state=random_state+i)
            return np.hstack([tp, fit])

        results = []
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            futures = [ex.submit(one, i) for i in range(n_subjects)]
            for f in tqdm(as_completed(futures), total=n_subjects, desc="Recovery", leave=False):
                results.append(f.result())

        cols = ["true_rho","true_lambda","true_delta","true_eta","true_beta",
                "fit_rho","fit_lambda","fit_delta","fit_eta","fit_beta"]
        df = pd.DataFrame(np.vstack(results), columns=cols)

        print("\n=== Parameter Recovery Correlations ===")
        for p in ["rho","lambda","delta","eta","beta"]:
            r = np.corrcoef(df[f"true_{p}"], df[f"fit_{p}"])[0,1]
            print(f"{p}: r = {r:.3f}")
        return df
