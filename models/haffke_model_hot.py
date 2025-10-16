import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import r2_score
from tqdm import trange


class HaffkeModel:
    """
    Optimized implementation of Haffke & Hübner (2020) Model 3 (CCT-hot).
    """

    def __init__(self, data_df, payoff_per_gain=30):
        self.df = data_df.copy().reset_index(drop=True)
        required = {'trial', 'state_n', 'gains_left', 'cards_left', 'choice', 'is_first_choice'}
        if not required.issubset(self.df.columns):
            raise ValueError(f"Dataframe must contain columns: {required}.")
        self.payoff = payoff_per_gain

    # ----- utility functions -----
    @staticmethod
    def value_function(x, alpha):
        return np.power(x, alpha)

    @staticmethod
    def prelec_weight(p, delta, eta):
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return np.exp(-delta * (-np.log(p)) ** eta)

    # ----- probability computations -----
    @staticmethod
    def p_conditional_vec(gains_left, cards_left):
        return np.clip(gains_left / cards_left, 1e-12, 1.0)

    @staticmethod
    def p_conjunctive_from_len_vec(g, c, length):
        """Vectorized conjunctive probability calculation (safe)."""
        if len(g) == 0:
            return np.array([], dtype=float)  # ⚡️ Если пустой — сразу выходим

        p = np.ones_like(g, dtype=float)
        mask = (length > 0) & (length <= c)
        valid_idx = np.where(mask)[0]
        if valid_idx.size == 0:
            return np.clip(p, 1e-12, 1.0)

        max_len = int(np.max(length[valid_idx]))
        for i in range(max_len):
            submask = valid_idx[length[valid_idx] > i]
            if len(submask) == 0:
                break
            numer = g[submask] - i
            denom = c[submask] - i
            p[submask] *= np.clip(numer / denom, 0, 1)
        return np.clip(p, 1e-12, 1.0)

    def compute_pn_vec(self, model='conjunctive_complex', conj_length_rule='state'):
        """Vectorized computation of p_n for all rows."""
        g = self.df['gains_left'].values
        c = self.df['cards_left'].values
        s = self.df['state_n'].values
        is_first = self.df['is_first_choice'].values.astype(bool)

        if model == 'conditional':
            return self.p_conditional_vec(g, c)

        elif model == 'conjunctive_simple':
            m = s if conj_length_rule == 'state' else np.full_like(s, conj_length_rule)
            return self.p_conjunctive_from_len_vec(g, c, m)

        elif model == 'conjunctive_complex':
            p = np.empty_like(g, dtype=float)
            m = s if conj_length_rule == 'state' else np.full_like(s, conj_length_rule)

            # отдельная обработка, если нет вторичных решений
            if np.all(is_first):
                p[:] = self.p_conditional_vec(g, c)
                return p

            p[is_first] = self.p_conditional_vec(g[is_first], c[is_first])
            p[~is_first] = self.p_conjunctive_from_len_vec(g[~is_first], c[~is_first], m[~is_first])
            return p

        else:
            raise ValueError("Unknown model type")


    # ----- vectorized probability of 'take' -----
    def fast_prob_take(self, params, model='conjunctive_complex', conj_length_rule='state'):  # ⚡
        alpha, delta, eta, theta = params
        n = self.df['state_n'].values
        x_n = n * self.payoff
        x_np1 = (n + 1) * self.payoff

        p_obj = self.compute_pn_vec(model=model, conj_length_rule=conj_length_rule)
        pi_p = self.prelec_weight(p_obj, delta, eta)

        V_take = self.value_function(x_np1, alpha) * pi_p
        V_stop = self.value_function(x_n, alpha)

        logits = theta * (V_take - V_stop)
        return np.clip(expit(logits), 1e-12, 1 - 1e-12)

    # ----- negative log-likelihood -----
    def neg_log_likelihood(self, params, model='conjunctive_complex', conj_length_rule='state'):
        alpha, delta, eta, theta = params
        if not (0 < alpha < 3 and 0 < delta < 3 and 0 < eta < 3 and 0 < theta <= 10):
            return 1e9
        p = self.fast_prob_take(params, model=model, conj_length_rule=conj_length_rule)
        choice = self.df['choice'].values
        # stable NLL
        nll = -np.sum(choice * np.log(p) + (1 - choice) * np.log(1 - p))
        return nll

    # ----- fit -----
    def fit(self, model='conjunctive_complex', conj_length_rule='state',
            n_starts=200, method='Nelder-Mead', rng_seed=123):
        rng = np.random.RandomState(rng_seed)
        best_val, best_res = np.inf, None

        for s in range(n_starts):
            start = [
                rng.uniform(0.05, 3.0),
                rng.uniform(0.05, 3.0),
                rng.uniform(0.05, 3.0),
                rng.uniform(0.1, 10.0)
            ]
            res = minimize(self.neg_log_likelihood, start,
                           args=(model, conj_length_rule),
                           method=method,
                           options={'maxiter': 800, 'disp': False})
            if res.success and res.fun < best_val:
                best_val, best_res = res.fun, res

        if best_res is None:
            raise RuntimeError("Model fit failed in all starting points.")
        self.fitted_params_ = best_res.x
        self.fitted_model_ = model             # ⚡ сохраняем тип модели
        self.conj_rule_ = conj_length_rule
        return best_res.x
    
    # ----- posterior predictive check -----
    def predictive_check(self, n_sims=100, seed=42):
        if not hasattr(self, 'fitted_params_'):
            raise ValueError("Model must be fitted first.")

        rng = np.random.RandomState(seed)
        sims = np.zeros((n_sims, len(self.df)), dtype=int)

        for s in range(n_sims):
            pvec = self.fast_prob_take(self.fitted_params_,
                                    model=self.fitted_model_,
                                    conj_length_rule=self.conj_rule_)
            sims[s, :] = rng.binomial(1, pvec)

        sim_mean = sims.mean(axis=0)
        obs = self.df['choice'].values.astype(float)
        r2 = r2_score(obs, sim_mean)

        states = sorted(self.df['state_n'].unique())
        obs_prop = {st: self.df.loc[self.df['state_n'] == st, 'choice'].mean() for st in states}
        sim_prop = {st: sim_mean[self.df['state_n'] == st].mean() for st in states}

        print(f"Predictive check R^2 (obs vs sim mean): {r2:.4f}")
        return {'r2': r2, 'obs_by_state': obs_prop, 'sim_by_state': sim_prop, 'sims': sims}

    # ----- parameter recovery -----
    def parameter_recovery(self, template_df, model='conjunctive_complex',
                        conj_length_rule='state', n_subjects=30, rng_seed=1):
        rng = np.random.RandomState(rng_seed)
        trues, fits = [], []

        for _ in trange(n_subjects, desc="Parameter recovery"):
            true = np.array([
                rng.uniform(0.05, 3.0),   # alpha
                rng.uniform(0.05, 3.0),   # delta
                rng.uniform(0.05, 3.0),   # eta
                rng.uniform(0.1, 10.0)    # theta
            ])

            temp_model = HaffkeModel(template_df, payoff_per_gain=self.payoff)
            pvec = temp_model.fast_prob_take(true, model=model, conj_length_rule=conj_length_rule)
            choices = rng.binomial(1, pvec)
            sim_df = template_df.copy()
            sim_df['choice'] = choices

            fit_model = HaffkeModel(sim_df, payoff_per_gain=self.payoff)
            try:
                fit_params = fit_model.fit(model=model,
                                        conj_length_rule=conj_length_rule,
                                        n_starts=200,
                                        rng_seed=rng.randint(0, 2**30))
            except Exception:
                fit_params = np.array([np.nan] * 4)

            trues.append(true)
            fits.append(fit_params)

        trues, fits = np.vstack(trues), np.vstack(fits)
        df = pd.DataFrame(np.hstack([trues, fits]),
                        columns=['true_alpha', 'true_delta', 'true_eta', 'true_theta',
                                'fit_alpha', 'fit_delta', 'fit_eta', 'fit_theta'])

        corrs = {p: np.corrcoef(df[f'true_{p}'], df[f'fit_{p}'])[0, 1]
                if not np.isnan(df[f'fit_{p}']).all() else np.nan
                for p in ['alpha', 'delta', 'eta', 'theta']}

        print("Parameter recovery correlations:", corrs)
        return {'df': df, 'correlations': corrs}
