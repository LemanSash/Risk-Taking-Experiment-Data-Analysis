import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import r2_score

class VPPModel:
    """
    Value-Plus-Perseveration (VPP) model implementation (Worthy et al., 2013).

    Параметры (в порядке вектора params):
      [phi, alpha, lam, c, w, K, eps_pos, eps_neg]

      phi     : learning rate for expected value (0..1)
      alpha   : outcome sensitivity (0.01..1.0)
      lam     : loss aversion (>0, e.g. [0.01, 5.0])
      c       : consistency parameter (0..5), transformed as θ = 3c - 1
      w       : weight on expected value vs perseveration (0..1)
      K       : perseveration decay (0..1)
      eps_pos : increment to perseveration when chosen outcome >= 0 ([-1,1])
      eps_neg : increment to perseveration when chosen outcome <  0 ([-1,1])

    Ожидаемые колонки входного DataFrame:
      'trial_number' (опционально), 'deck_num' (0..3), 'points_earned' (float)
    """

    def __init__(self, data_df=None):
        self.data = None if data_df is None else data_df.copy()

    @staticmethod
    def _utility(x, alpha, lam):
        """Prospect-like utility function."""
        x = float(x)
        eps = 1e-12
        if x >= 0.0:
            return (x + eps) ** float(alpha)
        else:
            return - float(lam) * ((-x + eps) ** float(alpha))

    @staticmethod
    def nll(params, data):
        """
        Negative log-likelihood for one subject/session.
        params: [phi, alpha, lam, c, w, K, eps_pos, eps_neg]
        """
        phi, alpha, lam, c, w, K, eps_pos, eps_neg = params

        # enforce ranges / stability
        phi = float(np.clip(phi, 0.0, 1.0))
        alpha = float(np.clip(alpha, 0.01, 1.0))
        lam = float(np.clip(lam, 0.01, 5.0))
        c = float(np.clip(c, 0.0, 5.0))
        theta = 3.0 * c - 1.0
        w = float(np.clip(w, 0.0, 1.0))
        K = float(np.clip(K, 0.0, 1.0))
        eps_pos = float(np.clip(eps_pos, -1.0, 1.0))
        eps_neg = float(np.clip(eps_neg, -1.0, 1.0))

        # initialize expected values and perseveration arrays
        E = np.zeros(4, dtype=float)
        P = np.zeros(4, dtype=float)
        nll = 0.0

        decks = data['deck_num'].to_numpy(dtype=int)
        rewards = data['points_earned'].to_numpy(dtype=float)

        for deck, reward in zip(decks, rewards):
            #deck = int(row['deck_num'])

            # composite value
            V = w * E + (1.0 - w) * P

            # numerically stable log-softmax
            v = theta * V
            v_max = np.max(v)
            log_denom = v_max + np.log(np.sum(np.exp(v - v_max)))
            logp = v[deck] - log_denom
            nll -= logp

            # outcome utility
            #reward = float(row['points_earned'])
            u = VPPModel._utility(reward, alpha, lam)

            # update E and P
            E[deck] = E[deck] + phi * (u - E[deck])
            P = K * P
            if reward >= 0:
                P[deck] += eps_pos
            else:
                P[deck] += eps_neg

        return float(nll)


    def fit(self, user_data, n_restarts=5, x0=None, bounds=None, random_seed=3):
        """
        Fit VPP to single subject using L-BFGS-B MLE.
        n_restarts: увеличить число рестартов (рекомендация 30-50)
        """
        if x0 is None:
            x0 = [0.3, 0.8, 1.0, 2.5, 0.7, 0.8, 0.2, -0.2]

        if bounds is None:
            bounds = [
                (0.0, 1.0),    # phi
                (0.01, 2.5),   # alpha (расширенный верх)
                (0.01, 10.0),  # lam   (расширенный)
                (0.0, 5.0),    # c
                (0.0, 1.0),    # w
                (0.0, 1.0),    # K
                (-2.0, 2.0),   # eps_pos (шире)
                (-2.0, 2.0)    # eps_neg (шире)
            ]

        data = user_data.copy().sort_values(by=['trial_number']).reset_index(drop=True)

        best_res = None
        best_fun = np.inf

        inits = [np.array(x0, dtype=float)]
        rng = np.random.RandomState(random_seed)
        for _ in range(max(0, n_restarts - 1)):
            init = np.array([rng.uniform(lo, hi) for (lo, hi) in bounds], dtype=float)
            inits.append(init)

        for start in inits:
            try:
                res = minimize(lambda p: VPPModel.nll(p, data), 
                               start, 
                               method='L-BFGS-B', 
                               bounds=bounds,
                               options={'maxiter': 150}
                               )
            except Exception:
                continue
            # если оптимизатор вернул nan/inf — пропускаем
            if (res is None) or (not hasattr(res, 'fun')):
                continue
            if np.isfinite(res.fun) and res.fun < best_fun:
                best_fun = float(res.fun)
                best_res = res

        if best_res is None:
            return np.array([np.nan]*8), np.nan

        params = np.array(best_res.x, dtype=float)

        # детекция выхода на границы (логирование)
        for i, ((lo, hi), val) in enumerate(zip(bounds, params)):
            if np.isclose(val, lo) or np.isclose(val, hi):
                # печать предупреждения (можно заменить логированием)
                print(f"Warning: param {i} at bound ({val:.4g} in [{lo},{hi}])")

        return params, best_fun

    
    def _build_empirical_payoff_dict(self):
        payoff_dict = {}
        if self.data is None:
            return None
        df = self.data.copy()
        if 'deck_num' not in df.columns or 'points_earned' not in df.columns:
            return None
        for d in sorted(df['deck_num'].unique()):
            payoffs = df.loc[df['deck_num'] == d, 'points_earned'].dropna().values
            if len(payoffs) > 0:
                payoff_dict[int(d)] = np.array(payoffs, dtype=float)
        # ensure all decks present
        for k in range(4):
            if k not in payoff_dict:
                payoff_dict[k] = None
        return payoff_dict

    def simulate(self, params, n_trials, seed=None, use_empirical=True):
        """
        Simulate n_trials using VPP parameters.
        If use_empirical and self.data exists, draw rewards from empirical payoff distribution per deck.
        Otherwise fallback to random (canonical or normal).
        """
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()

        phi, alpha, lam, c, w, K, eps_pos, eps_neg = params
        phi = float(np.clip(phi, 0.0, 1.0))
        alpha = float(np.clip(alpha, 0.01, 2.5))   # расширенный верхний предел
        lam = float(np.clip(lam, 0.01, 10.0))      # расширенный
        c = float(np.clip(c, 0.0, 5.0))
        theta = 3.0 * c - 1.0
        w = float(np.clip(w, 0.0, 1.0))
        K = float(np.clip(K, 0.0, 1.0))
        eps_pos = float(eps_pos)
        eps_neg = float(eps_neg)

        E = np.zeros(4, dtype=float)
        P = np.zeros(4, dtype=float)
        payoff_dict = None
        if use_empirical:
            payoff_dict = self._build_empirical_payoff_dict()

        sim_rows = []
        for t in range(1, n_trials + 1):
            V = w * E + (1.0 - w) * P
            v = theta * V
            v = v - np.max(v)
            ex = np.exp(v)
            probs = ex / (np.sum(ex) + 1e-16)
            deck = int(rng.choice(4, p=probs))

            # choose reward
            reward = None
            if payoff_dict is not None and payoff_dict.get(deck) is not None and len(payoff_dict[deck]) > 0:
                reward = float(rng.choice(payoff_dict[deck]))
            else:
                # fallback: moderate-variance random payoff
                # пример: центр 100 для decks 0/1 и 50 для 2/3, sd=40 (меньше чем раньше)
                reward = float(rng.normal(loc=100.0 if deck in [0,1] else 50.0, scale=40.0))

            sim_rows.append({'trial_number': t, 'deck_num': deck, 'points_earned': reward})

            u = VPPModel._utility(reward, alpha, lam)
            E[deck] = E[deck] + phi * (u - E[deck])
            P = K * P
            if reward >= 0:
                P[deck] += eps_pos
            else:
                P[deck] += eps_neg

        return pd.DataFrame(sim_rows)


    def predictive_check(self, real_data, seed=42):
        params, _ = self.fit(real_data)
        #sim = self.simulate(params, n_trials=len(real_data), seed=seed)
        sim = self.simulate(params, n_trials=len(real_data), seed=seed, use_empirical=True)

        def adv_rates(df, n_blocks=5):
            df2 = df.sort_values('trial_number').reset_index(drop=True)
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
                rate = np.mean(block['deck_num'].isin([2,3]))
                rates.append(rate)
            return np.array(rates, dtype=float)

        real_vec = adv_rates(real_data)
        sim_vec = adv_rates(sim)
        r2 = r2_score(real_vec, sim_vec)
        diag = pd.DataFrame({'block': np.arange(1, len(real_vec)+1),
                             'real_rate': real_vec, 'sim_rate': sim_vec})
        print(diag)
        print(f"Fitted params: {params}")
        print(f"R² between real and simulated block rates: {r2:.4f}")
        return r2, params, diag

    def parameter_recovery(self, n_subjects=40, n_trials=200, seed0=300):
        rng = np.random.RandomState(seed0)
        true_params = []
        recovered = []

        for i in range(n_subjects):
            p_true = [
                rng.rand(),                # phi
                rng.uniform(0.01, 1.0),    # alpha
                rng.uniform(0.1, 5.0),     # lam
                rng.uniform(0.5, 4.5),     # c
                rng.rand(),                # w
                rng.rand(),                # K
                rng.uniform(-1.0, 1.0),    # eps_pos
                rng.uniform(-1.0, 1.0)     # eps_neg
            ]
            sim = self.simulate(p_true, n_trials=n_trials, seed=seed0+i)
            p_fit, _ = self.fit(sim, n_restarts=5)
            true_params.append(p_true)
            recovered.append(p_fit)

        true_arr = np.array(true_params)
        rec_arr = np.array(recovered)
        cols = ['phi','alpha','lam','c','w','K','eps_pos','eps_neg']
        df = pd.DataFrame(
            np.hstack([true_arr, rec_arr]),
            columns=[f"true_{c}" for c in cols] + [f"fit_{c}" for c in cols]
        )

        print("Parameter Recovery: correlations (true vs fit)")
        for c in cols:
            if np.nanstd(df[f'fit_{c}']) == 0:
                corr = np.nan
            else:
                corr = np.corrcoef(df[f'true_{c}'], df[f'fit_{c}'])[0,1]
            print(f"  {c}: r = {np.round(corr,3) if not np.isnan(corr) else 'nan'}")
        return df
