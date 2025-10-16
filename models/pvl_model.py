import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import r2_score

class PVLModel:
    """
    PVL model with Delta/Decay rules, TIC mapping theta = 3**c - 1 (TIC),
    and reward scaling to stabilize estimation for large-magnitude IGT payoffs.
    """

    def __init__(self, data_df=None, update_rule="delta", reward_scale=100.0):
        self.data = None if data_df is None else data_df.copy()
        self.update_rule = update_rule.lower()
        assert self.update_rule in ["delta", "decay"], "update_rule must be 'delta' or 'decay'"
        self.reward_scale = float(reward_scale)

    @staticmethod
    def _utility(x, alpha, lam):
        x = float(x)
        eps = 1e-12
        if x >= 0:
            return (x + eps) ** float(alpha)
        else:
            return - float(lam) * ((-x + eps) ** float(alpha))

    @staticmethod
    def _theta_from_c(c, theta_clip=100.0):
        theta = 3.0 ** float(c) - 1.0
        # clip to avoid numeric overflow / extreme determinism
        if not np.isfinite(theta):
            theta = theta_clip
        return float(np.minimum(theta, theta_clip))

    @staticmethod
    def _safe_softmax(values, c, theta_clip=100.0):
        theta = PVLModel._theta_from_c(c, theta_clip=theta_clip)
        v = np.asarray(values, dtype=float) * theta
        v = v - np.max(v)
        ex = np.exp(v)
        probs = ex / (np.sum(ex) + 1e-16)
        return probs

    @staticmethod
    def nll(params, data, update_rule="delta", reward_scale=100.0, theta_clip=100.0):
        # params: [A, alpha, lam, c]
        A, alpha, lam, c = params
        A = np.clip(A, 0.0, 1.0)
        alpha = np.clip(alpha, 0.01, 1.0)   # prevent alpha -> 0
        lam = max(lam, 1e-6)
        c = np.clip(c, 0.0, 3.0)           # narrower c range

        Q = np.zeros(4, dtype=float)
        nll = 0.0

        for idx, row in data.reset_index(drop=True).iterrows():
            deck = int(row['deck_num'])
            probs = PVLModel._safe_softmax(Q, c, theta_clip=theta_clip)
            p_chosen = np.clip(probs[deck], 1e-12, 1.0)
            nll -= np.log(p_chosen)

            reward_raw = float(row['points_earned'])
            reward = reward_raw / float(reward_scale)  # scale input reward
            u = PVLModel._utility(reward, alpha, lam)

            if update_rule == "delta":
                Q[deck] = Q[deck] + A * (u - Q[deck])
            else:  # decay
                Q = A * Q
                Q[deck] += u

        return float(nll)

    def fit(self, user_data, n_restarts=24, x0=None, bounds=None, theta_clip=100.0):
        if x0 is None:
            x0 = [0.3, 0.3, 1.0, 1.0]
        if bounds is None:
            bounds = [
                (0.0, 1.0),    # A
                (0.01, 1.0),   # alpha (no zero)
                (0.01, 10.0),  # lam
                (0.0, 3.0)     # c
            ]

        data = user_data.copy().sort_values(by=['trial_number']).reset_index(drop=True)

        best_res = None
        best_fun = np.inf

        rng = np.random.RandomState(12345)
        inits = [np.array(x0, dtype=float)]
        for _ in range(max(0, n_restarts - 1)):
            init = np.array([rng.uniform(lo, hi) for (lo, hi) in bounds], dtype=float)
            inits.append(init)

        for start in inits:
            try:
                res = minimize(lambda p: PVLModel.nll(p, data, self.update_rule, self.reward_scale, theta_clip),
                               start, method='L-BFGS-B', bounds=bounds)
            except Exception:
                continue
            if res.success and res.fun < best_fun:
                best_fun = float(res.fun)
                best_res = res

        if best_res is None:
            return np.array([np.nan, np.nan, np.nan, np.nan]), np.nan

        params = np.array(best_res.x, dtype=float)
        # final clipping
        params[0] = float(np.clip(params[0], 0.0, 1.0))
        params[1] = float(np.clip(params[1], 0.01, 1.0))
        params[2] = float(max(params[2], 1e-6))
        params[3] = float(np.clip(params[3], 0.0, 3.0))
        return params, best_fun

    def simulate(self, params, n_trials, seed=None, theta_clip=100.0):
        rng = np.random.RandomState(seed)
        A, alpha, lam, c = params
        A = np.clip(A, 0.0, 1.0)
        alpha = np.clip(alpha, 0.01, 1.0)
        lam = max(lam, 1e-6)
        c = np.clip(c, 0.0, 3.0)

        Q = np.zeros(4, dtype=float)
        sim_rows = []

        for t in range(1, n_trials + 1):
            probs = PVLModel._safe_softmax(Q, c, theta_clip=theta_clip)
            deck = int(rng.choice(4, p=probs))

            # canonical IGT-like payoffs (same large-scale values as original IGT)
            if deck in [0, 1]:
                choices = np.array([100.0]*8 + [-150.0, -250.0])
            else:
                choices = np.array([50.0]*8 + [-50.0, -100.0])
            reward_raw = float(rng.choice(choices))
            reward = reward_raw / float(self.reward_scale)  # same scaling as in fit

            sim_rows.append({'trial_number': t, 'deck_num': deck, 'points_earned': reward_raw})
            u = PVLModel._utility(reward, alpha, lam)

            if self.update_rule == "delta":
                Q[deck] = Q[deck] + A * (u - Q[deck])
            else:
                Q = A * Q
                Q[deck] += u

        return pd.DataFrame(sim_rows)

    def predictive_check(self, real_data, seed=42, theta_clip=100.0):
        params, nll = self.fit(real_data)
        sim = self.simulate(params, n_trials=len(real_data), seed=seed, theta_clip=theta_clip)

        def adv_rates(df, n_blocks=5):
            df2 = df.copy().sort_values('trial_number').reset_index(drop=True)
            n = len(df2)
            block_size = int(np.ceil(n / n_blocks))
            rates = []
            for b in range(n_blocks):
                start = b * block_size
                end = min((b + 1) * block_size, n)
                if start >= end:
                    rates.append(0.0)
                    continue
                block = df2.iloc[start:end]
                rate = np.mean(block['deck_num'].isin([2, 3]))
                rates.append(rate)
            return np.array(rates, dtype=float)

        real_vec = adv_rates(real_data)
        sim_vec = adv_rates(sim)
        r2 = r2_score(real_vec, sim_vec)

        diag = pd.DataFrame({
            'block': np.arange(1, len(real_vec)+1),
            'real_rate': real_vec,
            'sim_rate': sim_vec
        })
        print(f"=== PVL-{self.update_rule.upper()} PPC: block-wise advantageous deck rates (decks 2 & 3) ===")
        print(diag)
        print(f"Fitted params: A={params[0]:.4f}, alpha={params[1]:.4f}, lam={params[2]:.4f}, c={params[3]:.4f}")
        print(f"R² between real and simulated block rates: {r2:.4f}")
        return r2, params, diag

    def parameter_recovery(self, n_subjects=50, n_trials=200, seed0=200):
        rng = np.random.RandomState(seed0)
        true_params = []
        recovered = []

        for i in range(n_subjects):
            p_true = [
                rng.rand(),             # A ∈ [0,1]
                rng.uniform(0.05, 0.9), # alpha ∈ [0.05,0.9]
                rng.uniform(0.2, 3.0),  # lam ∈ [0.2,3]
                rng.uniform(0.0, 3.0)   # c ∈ [0,3]
            ]
            sim = self.simulate(p_true, n_trials=n_trials, seed=seed0 + i)
            p_fit, _ = self.fit(sim, n_restarts=12)
            true_params.append(p_true)
            recovered.append(p_fit)

        true_arr = np.array(true_params)
        rec_arr = np.array(recovered)
        cols = ['A','alpha','lam','c']
        df = pd.DataFrame(
            np.hstack([true_arr, rec_arr]),
            columns=[f"true_{c}" for c in cols] + [f"fit_{c}" for c in cols]
        )

        print(f"Parameter Recovery ({self.update_rule.upper()}): correlations (true vs fit)")
        for c in cols:
            if np.nanstd(df[f'fit_{c}']) == 0:
                corr = np.nan
            else:
                corr = np.corrcoef(df[f'true_{c}'], df[f'fit_{c}'])[0,1]
            print(f"  {c}: r = {np.round(corr,3) if not np.isnan(corr) else 'nan'}")
        return df
