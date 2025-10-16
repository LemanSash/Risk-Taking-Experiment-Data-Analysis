import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import r2_score

class EVModel:
    """
    Expectancy-Valence (EV) model for the Iowa Gambling Task (Busemeyer & Stout, 2002).

    Parameters (free parameters to fit):
      - w : loss-weight (0..1). Attentional weight to losses vs wins.
      - a : learning / updating rate (0..1). How much new valence updates deck expectancy.
      - c : sensitivity exponent (real). Controls how sensitivity theta(t) = (t/10)^c evolves over trials.
            (c can be negative -> decreasing sensitivity, or positive -> increasing sensitivity).

    Data format expected (same as your VSEModel):
      DataFrame with columns:
        - 'trial_number' (1-based int) optional (if missing we use sequential index)
        - 'deck_num' (0..3)
        - 'points_earned' (net reward for that trial; can be + or -)

    Methods:
      - nll(params, data): negative log likelihood for one subject/session
      - fit(user_data, n_restarts=10, ...): fit model to one user's data (returns best_params, best_nll)
      - simulate(params, n_trials, seed=None): simulate a subject with given params
      - predictive_check(real_data, seed=42): simple PPC comparing advantageous deck rates (like VSE PPC)
      - parameter_recovery(...): simulate-and-recover pipeline
    """

    def __init__(self, data_df=None):
        self.data = None if data_df is None else data_df.copy()

    @staticmethod
    def _safe_softmax(ev_values, theta):
        """
        Numerically stable softmax for values scaled by theta:
          probs_j = exp(theta * ev_j) / sum_k exp(theta * ev_k)
        We clip/shift by max to avoid overflow/underflow.
        """
        v = np.asarray(ev_values, dtype=float) * float(theta)
        v = v - np.max(v)
        ex = np.exp(v)
        probs = ex / (np.sum(ex) + 1e-16)
        return probs

    @staticmethod
    def _compute_valence(reward, w):
        """
        Compute valence according to EV model, with normalized rewards.
        Scaling rewards by 1/100 maintains the model’s form but prevents overflow.
        """
        reward = float(reward)
        R = reward if reward > 0 else 0.0
        L = reward if reward < 0 else 0.0
        # (1) scale reward magnitudes for numerical stability
        R, L = R / 10.0, L / 10.0
        v = (1.0 - w) * R + w * L
        return v

    @staticmethod
    def _theta_from_c(trial_index, c, clip_theta=(1e-8, 1e8)):
        """
        theta(t) = (t/10)^c  (original EV formulation).
        We accept trial_index (1-based). Clip theta for numeric stability.
        """
        base = float(trial_index) / 10.0

        if base <= 0.0:
            theta = 0.0
        else:
            theta = base ** float(c)

        lo, hi = clip_theta
        theta = float(np.clip(theta, lo, hi))
        return theta

    @staticmethod
    def nll(params, data):
        """
        Negative log likelihood for EV model on a single subject/session.

        params: [w, a, c]
          w in [0,1]
          a in [0,1]
          c in [-inf, +inf] (we will clip in fit)

        data: DataFrame sorted by trial_number, with columns 'deck_num' and 'points_earned'.
        """
        w, a, c = params
        w = np.clip(w, 0.0, 1.0)
        a = np.clip(a, 0.0, 1.0)
        c = float(np.clip(c, -3, 3))

        Ev = np.zeros(4, dtype=float)   # Ev_j(0) = 0
        nll = 0.0

        if 'trial_number' in data.columns:
            trial_numbers = data['trial_number'].astype(int).values
        else:
            trial_numbers = np.arange(1, len(data) + 1)

        for idx, row in data.reset_index(drop=True).iterrows():
            deck = int(row['deck_num'])
            t = int(trial_numbers[idx])
            theta = EVModel._theta_from_c(t, c)

            # compute choice probabilities from Ev via softmax
            probs = EVModel._safe_softmax(Ev, theta)
            p_chosen = probs[deck]
            p_chosen = np.clip(p_chosen, 1e-12, 1.0)
            nll -= np.log(p_chosen)

            # compute valence from observed reward
            reward = float(row['points_earned'])
            v = EVModel._compute_valence(reward, w)

            # update only chosen deck expectancy:
            # Ev_j(t) = (1-a) * Ev_j(t-1) + a * v
            Ev[deck] = (1.0 - a) * Ev[deck] + a * v

        return float(nll)

    def fit(self, user_data, n_restarts=10, x0=None, bounds=None):
        """
        Fit EV model to one user's data (MLE via L-BFGS-B).
        Returns: params_array [w,a,c], best_nll

        user_data: DataFrame with ['trial_number','deck_num','points_earned'] sorted by trial_number
        """
        if x0 is None:
            x0 = [0.35, 0.3, 0.3]
        if bounds is None:
            bounds = [(0.0, 1.0),    # w
                      (0.0, 1.0),    # a
                      (-3, 3)]       # c 

        data = user_data.copy().sort_values(by=['trial_number']).reset_index(drop=True)

        best_res = None
        best_fun = np.inf

        # generate restarts (one deterministic + random restarts)
        inits = [np.array(x0, dtype=float)]
        rng = np.random.RandomState(1)
        for _ in range(max(0, n_restarts - 1)):
            init = np.array([rng.uniform(lo, hi) for (lo, hi) in bounds], dtype=float)
            inits.append(init)

        for start in inits:
            try:
                res = minimize(lambda p: EVModel.nll(p, data), start, method='L-BFGS-B', bounds=bounds)
            except Exception:
                continue
            if res.success and res.fun < best_fun:
                best_fun = float(res.fun)
                best_res = res

        if best_res is None:
            try:
                res = minimize(lambda p: EVModel.nll(p, data), x0, method='L-BFGS-B', bounds=bounds)
                best_res = res
                best_fun = float(res.fun)
            except Exception:
                return np.array([np.nan, np.nan, np.nan]), np.nan

        params = np.array(best_res.x, dtype=float)
        params[0] = np.clip(params[0], 0.0, 1.0)  # w
        params[1] = np.clip(params[1], 0.0, 1.0)  # a
        params[2] = float(np.clip(params[2], bounds[2][0], bounds[2][1]))  # c
        return params, best_fun

    def _build_empirical_payoff_dict(self):
        """
        Build dict deck -> observed empirical payoffs from self.data (if provided);
        Used by simulate() to sample realistic rewards.
        """
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
        for k in range(4):
            if k not in payoff_dict:
                payoff_dict[k] = None
        return payoff_dict

    def simulate(self, params, n_trials, seed=None):
        """
        Simulate a subject for n_trials using EV model and either empirical payoffs
        (if self.data provided) or canonical payoffs matching typical IGT structure.

        params: [w, a, c]
        Returns DataFrame with columns: 'trial_number','deck_num','points_earned'
        """
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()

        w, a, c = params
        Ev = np.zeros(4, dtype=float)
        payoff_dict = self._build_empirical_payoff_dict()

        sim_rows = []
        for t in range(1, n_trials + 1):
            theta = EVModel._theta_from_c(t, c)
            probs = EVModel._safe_softmax(Ev, theta)

            deck = int(rng.choice(4, p=probs))

            # pick reward: from empirical if available, else canonical IGT-like draws
            if payoff_dict is not None and payoff_dict.get(deck) is not None and len(payoff_dict[deck]) > 0:
                reward = float(rng.choice(payoff_dict[deck]))
            else:
                if deck in [0, 1]:  # disadvantageous decks (A, B)
                    choices = np.array([100.0]*8 + [-150.0, -250.0])
                    reward = float(rng.choice(choices))
                else:               # advantageous decks (C, D)
                    choices = np.array([50.0]*8 + [-50.0, -100.0])
                    reward = float(rng.choice(choices))

            sim_rows.append({'trial_number': t, 'deck_num': deck, 'points_earned': reward})

            v = EVModel._compute_valence(reward, w)
            Ev[deck] = (1.0 - a) * Ev[deck] + a * v

        return pd.DataFrame(sim_rows)

    def predictive_check(self, real_data, seed=42):
        """
        Simple posterior predictive check:
        - fit params on real_data
        - simulate one subject with fitted params for same n_trials
        - compare proportion of selections from advantageous decks (2 & 3) in 5-blocks
        - compute R^2 between real and simulated block-wise advantageous rates
        """
        params, nll = self.fit(real_data, n_restarts=12)
        sim = self.simulate(params, n_trials=len(real_data), seed=seed)

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
            'block': np.arange(1, len(real_vec) + 1),
            'real_rate': real_vec,
            'sim_rate': sim_vec
        })

        print("=== EV PPC ===")
        print(diag)
        print(f"Fitted params: w={params[0]:.4f}, a={params[1]:.4f}, c={params[2]:.4f}")
        print(f"R² between real and simulated block rates: {r2:.4f}")

        return r2, params, diag


    def parameter_recovery(self, n_subjects=50, n_trials=200, seed0=100):
        """
        Parameter recovery routine:
         - for n_subjects: sample random true params, simulate data, fit model back
         - return DataFrame with true vs fit params
        """
        rng = np.random.RandomState(seed0)
        true_params = []
        recovered = []

        for i in range(n_subjects):
            p_true = [
                rng.rand(),            # w ~ U(0,1)
                rng.rand(),            # a ~ U(0,1)
                rng.uniform(-2.5, 2.5) # c ~ U(-2.5, 2.5)
            ]
            sim = self.simulate(p_true, n_trials=n_trials, seed=seed0 + i)
            p_fit, _ = self.fit(sim, n_restarts=8)
            true_params.append(p_true)
            recovered.append(p_fit)

        true_arr = np.array(true_params)
        rec_arr = np.array(recovered)
        cols = ['w','a','c']
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
