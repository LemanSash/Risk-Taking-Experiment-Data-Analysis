import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import r2_score
from tqdm import tqdm
from numba import njit

# ---------- JIT HELPERS ----------

@njit
def p_gain_array_numba(N, L):
    """P(no loss in k draws without replacement)"""
    p = np.zeros(N + 1)
    p[0] = 1.0
    M = N - L
    if M <= 0:
        return p
    max_k = min(N, M)
    logs = np.zeros(max_k)
    for i in range(max_k):
        logs[i] = np.log((M - i) / (N - i))
    cumsum = 0.0
    for k in range(max_k):
        cumsum += logs[k]
        p[k + 1] = np.exp(cumsum)
    return p

@njit
def logsumexp_numba(arr):
    m = np.max(arr)
    s = 0.0
    for i in range(arr.shape[0]):
        s += np.exp(arr[i] - m)
    return m + np.log(s)

@njit
def trial_nll_numba(N, L, g, l, k_obs, rho, lam, delta, eta, beta):
    """One-trial negative log-likelihood (fast JIT)."""
    p_gain = p_gain_array_numba(N, L)
    p_loss = 1.0 - p_gain
    EU = np.empty(N + 1)
    for k in range(N + 1):
        xg = k * g
        ug = abs(xg) ** rho
        if xg < 0:
            ug *= -lam
        ul = abs(l) ** rho
        if l < 0:
            ul *= -lam
        pg = min(max(p_gain[k], 1e-12), 1.0 - 1e-12)
        pl = min(max(p_loss[k], 1e-12), 1.0 - 1e-12)
        wg = np.exp(-delta * ((-np.log(pg)) ** eta))
        wl = np.exp(-delta * ((-np.log(pl)) ** eta))
        EU[k] = wg * ug + wl * ul
    z = EU / beta
    lse = logsumexp_numba(z)
    return - (z[k_obs] - lse)


# ---------- MAIN CLASS ----------

class COLDModel2:
    """Cold Columbia Card Task – Model 2 (Prospect Theory + Prelec weighting) with Numba acceleration."""

    def __init__(self, data_df):
        self.data = data_df.copy()

    # ---------- negative log-likelihood ----------
    def nll(self, params, df):
        rho, lam, delta, eta, beta = params
        if np.any(np.array(params) <= 0) or not np.isfinite(params).all():
            return 1e9
        nll = 0.0
        for _, tr in df.iterrows():
            N = int(tr.get("total_cards", 32))
            L = int(tr["loss_cards"])
            g = float(tr["gain_amount"])
            l = -abs(float(tr["loss_amount"]))
            k_obs = int(tr["num_cards"])
            if k_obs < 0 or k_obs > N:
                return 1e9
            nll += trial_nll_numba(N, L, g, l, k_obs, rho, lam, delta, eta, beta)
            if nll > 1e8:
                return 1e9
        return float(nll)

    # ---------- fit ----------
    def fit(self, df, bounds=None, n_starts=10, n_jobs=4, random_state=None):
        if bounds is None:
            bounds = [(0.01,3.0),(0.01,10.0),(0.01,5.0),(0.01,3.0),(0.01,1.0)]
        rng = np.random.default_rng(random_state)

        def one_start(_):
            x0 = [rng.uniform(lo, hi) for lo, hi in bounds]
            try:
                res = minimize(self.nll, x0, args=(df,), bounds=bounds, method="L-BFGS-B")
                if res is not None and np.isfinite(res.fun):
                    return res.fun, res.x
            except Exception:
                pass
            return np.inf, None

        best_fun, best_x = np.inf, None
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            futures = [ex.submit(one_start, i) for i in range(n_starts)]
            for f in tqdm(as_completed(futures), total=n_starts, desc="Multistart fit", leave=False):
                fun, x = f.result()
                if fun < best_fun:
                    best_fun, best_x = fun, x
        return best_x if best_x is not None else [np.nan]*5

    # ---------- simulate ----------
    def simulate(self, params, df, seed=None):
        rho, lam, delta, eta, beta = params
        rng = np.random.default_rng(seed)
        sim = df.copy().reset_index(drop=True)
        chosen_ks = []
        for _, tr in sim.iterrows():
            N = int(tr.get("total_cards", 32))
            L = int(tr["loss_cards"])
            g = float(tr["gain_amount"])
            l = -abs(float(tr["loss_amount"]))
            p_gain = p_gain_array_numba(N, L)
            p_loss = 1.0 - p_gain
            EU = np.zeros(N + 1)
            for k in range(N + 1):
                xg = k * g
                ug = abs(xg) ** rho
                if xg < 0:
                    ug *= -lam
                ul = abs(l) ** rho
                if l < 0:
                    ul *= -lam
                wg = np.exp(-delta * ((-np.log(np.clip(p_gain[k],1e-12,1-1e-12))) ** eta))
                wl = np.exp(-delta * ((-np.log(np.clip(p_loss[k],1e-12,1-1e-12))) ** eta))
                EU[k] = wg * ug + wl * ul
            z = EU / beta
            probs = np.exp(z - logsumexp(z))
            probs = np.clip(probs, 0, 1)
            probs /= probs.sum()
            chosen_ks.append(int(rng.choice(np.arange(N + 1), p=probs)))
        sim["num_cards_sim"] = chosen_ks
        return sim

    # ---------- predictive check ----------
    def predictive_check(self, df, fit_kwargs=None, seed=42):
        fit_kwargs = fit_kwargs or {}
        params = self.fit(df, **fit_kwargs)
        sim = self.simulate(params, df, seed)
        r2 = r2_score(df["num_cards"], sim["num_cards_sim"])
        print(f"PPC: ρ={params[0]:.3f}, λ={params[1]:.3f}, δ={params[2]:.3f}, η={params[3]:.3f}, β={params[4]:.3f}, R²={r2:.3f}")
        return r2

    # ---------- parameter recovery ----------
    def parameter_recovery(self, template_df, n_subjects=20, n_starts=10, n_jobs=8, random_state=123):
        """
        Simulate synthetic subjects with random true parameters,
        fit them back, and compute recovery correlations.
        """
        rng = np.random.default_rng(random_state)
        bounds = [(0.01,3.0),(0.01,10.0),(0.01,5.0),(0.01,3.0),(0.01,1.0)]
        true_params = np.column_stack([rng.uniform(lo, hi, n_subjects) for lo, hi in bounds])

        def one(i):
            tp = true_params[i]
            sim = self.simulate(tp, template_df, seed=100 + i)
            fit = self.fit(sim, bounds=bounds, n_starts=n_starts, n_jobs=4, random_state=random_state + i)
            return np.hstack([tp, fit])

        results = []
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            futures = [ex.submit(one, i) for i in range(n_subjects)]
            for f in tqdm(as_completed(futures), total=n_subjects, desc="Parameter Recovery", leave=False):
                results.append(f.result())

        cols = ["true_rho","true_lambda","true_delta","true_eta","true_beta",
                "fit_rho","fit_lambda","fit_delta","fit_eta","fit_beta"]
        df = pd.DataFrame(np.vstack(results), columns=cols)

        print("\n=== Parameter Recovery Correlations ===")
        for p in ["rho","lambda","delta","eta","beta"]:
            r = np.corrcoef(df[f"true_{p}"], df[f"fit_{p}"])[0,1]
            print(f"{p}: r = {r:.3f}")
        return df

