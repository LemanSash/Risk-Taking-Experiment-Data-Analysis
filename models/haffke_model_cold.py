import numpy as np
import pandas as pd
from math import lgamma
from scipy.optimize import minimize
from numba import njit
from sklearn.metrics import r2_score
from tqdm import trange

# ---------- вспомогательные функции ----------
@njit
def log_comb(n, k):
    if k > n or k < 0:
        return -np.inf
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)

@njit
def prelec_weight(p, delta, eta):
    # безопасная граница
    if p < 1e-12:
        p = 1e-12
    if p > 1.0:
        p = 1.0
    return np.exp(-delta * ((-np.log(p)) ** eta))

@njit
def value_func_pos(x, alpha):
    return max(x, 0.0) ** alpha

@njit
def softmax(vals, theta):
    max_val = np.max(vals)
    exp_vals = np.exp(theta * (vals - max_val))
    s = exp_vals.sum()
    if s == 0.0:
        return np.ones_like(vals) / len(vals)
    return exp_vals / s

# ---------- изменённый compute_option_value ----------
# (1) считаем биномиальную вероятность успеха для всех m карт сразу,
# (2) применяем Prelec к этой единой вероятности (вместо произведения Prelec по картам).
@njit
def compute_option_value(m, c, g, gain_amt, alpha, delta, eta):
    # если m > c или m > g тогда вероятность успеха = 0 (невозможно открыть m выигрышных)
    if m > c or m < 0:
        return 0.0
    # биномиальная вероятность, что все m выбранных карт будут выигрышными:
    # p_success = C(g, m) / C(c, m)
    if m == 0:
        p_success = 1.0
    else:
        if g < m:
            p_success = 0.0
        else:
            logp = log_comb(g, m) - log_comb(c, m)
            p_success = np.exp(logp)
    # применяем prelec только к итоговой p_success
    w_p = prelec_weight(p_success, delta, eta)
    EV = value_func_pos(m * gain_amt, alpha) * w_p
    return EV

@njit
def prob_choose_options(options, c, g, gain_amt, alpha, delta, eta, theta, lapse):
    n_opts = len(options)
    vals = np.zeros(n_opts)
    for j in range(n_opts):
        vals[j] = compute_option_value(options[j], c, g, gain_amt, alpha, delta, eta)
    probs = softmax(vals, theta)
    # добавляем lapse: небольшая вероятность равновероятного ответа
    if lapse <= 0.0:
        return probs
    unif = np.ones(n_opts) / n_opts
    final = (1.0 - lapse) * probs + lapse * unif
    return final

@njit
def predict_proba_numba(df_cards, df_gains, df_gain_amt, options_mat, options_len, params):
    alpha, delta, eta, theta, lapse = params
    n_trials = len(df_cards)
    max_opts = options_mat.shape[1]
    probs_matrix = np.zeros((n_trials, max_opts))
    
    for t in range(n_trials):
        n_opts = options_len[t]
        c = df_cards[t]
        g = df_gains[t]
        gain_amt = df_gain_amt[t]
        opts = options_mat[t, :n_opts]
        probs = prob_choose_options(opts, c, g, gain_amt, alpha, delta, eta, theta, lapse)
        probs_matrix[t, :n_opts] = probs
    return probs_matrix

# ---------- класс модели (с lapse) ----------
class HaffkeColdModel_Model3:
    def __init__(self, df):
        # df must contain: 'gains_left','cards_left','gain_amount','options','choice'
        self.df = df.copy().reset_index(drop=True)

    def neg_log_likelihood(self, theta):
        # theta = [alpha, delta, eta, theta_choice, lapse]
        alpha, delta, eta, theta_choice, lapse = theta
        # границы: небольшая положительная нижняя граница для stability, lapse в [0,0.2]
        if not (0.01 <= alpha <= 4 and 0.001 <= delta <= 4 and 0.001 <= eta <= 4 and 0.01 <= theta_choice <= 30 and 0.0 <= lapse <= 0.2):
            return 1e9

        nll = 0.0
        # подготовка опций для ускорения (реализовать в высокоуровневой функции при необходимости)
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            opts = np.array(row['options'], dtype=np.int64)
            probs = prob_choose_options(
                opts, int(row['cards_left']), int(row['gains_left']), float(row['gain_amount']),
                alpha, delta, eta, theta_choice, lapse
            )
            # защита - если probs невалидные
            if np.isnan(probs).any() or probs.sum() <= 0:
                probs = np.ones_like(probs) / len(probs)
            chosen_idx_arr = np.where(opts == int(row['choice']))[0]
            if chosen_idx_arr.size == 0:
                # если выбор не в опциях (неожиданно), даём маленькую вероятность
                p_chosen = 1e-12
            else:
                chosen_idx = chosen_idx_arr[0]
                p_chosen = max(probs[chosen_idx], 1e-12)
            nll -= np.log(p_chosen)
        return nll

    def fit(self, n_starts=30, method='L-BFGS-B', random_seed=0, verbose=True):
        rng = np.random.RandomState(random_seed)
        best_val, best_res = np.inf, None
        bounds = [
            (0.01, 4.0),    # alpha
            (0.001, 4.0),   # delta
            (0.001, 4.0),   # eta
            (0.01, 30.0),   # theta (inv temp)
            (0.0, 0.2)      # lapse
        ]
        for s in range(n_starts):
            start = np.array([
                rng.uniform(0.6, 1.6),   # alpha: ближе к 1 для стабильности
                rng.uniform(0.05, 1.5),  # delta
                rng.uniform(0.2, 2.0),   # eta
                rng.uniform(0.5, 10.0),  # theta
                rng.uniform(0.0, 0.08)   # lapse (небольшая)
            ])
            res = minimize(self.neg_log_likelihood, start, method=method, bounds=bounds)
            if res.success and res.fun < best_val:
                best_val, best_res = res.fun, res
        if best_res is None:
            raise RuntimeError("Optimization failed.")
        self.fitted_ = best_res.x
        if verbose:
            print(f"Best parameters: {self.fitted_}")
        return best_res

    def predict_proba(self, params=None):
        if params is None:
            if not hasattr(self, 'fitted_'):
                raise ValueError("Fit model first or pass params.")
            params = self.fitted_
        params = np.array(params, dtype=np.float64)
        if params.size != 5:
            raise ValueError("params must be length 5: [alpha, delta, eta, theta, lapse]")

        df_gains = self.df['gains_left'].values.astype(np.int64)
        df_cards = self.df['cards_left'].values.astype(np.int64)
        df_gain_amt = self.df['gain_amount'].values.astype(np.float64)
        df_options = self.df['options'].values

        max_len = max(len(o) for o in df_options)
        options_mat = np.zeros((len(df_options), max_len), dtype=np.int64)
        options_len = np.zeros(len(df_options), dtype=np.int64)
        for i, opts in enumerate(df_options):
            options_mat[i, :len(opts)] = np.array(opts, dtype=np.int64)
            options_len[i] = len(opts)

        return predict_proba_numba(df_cards, df_gains, df_gain_amt, options_mat, options_len, params)

    def predictive_check(self, n_sims=500, seed=0):
        if not hasattr(self, 'fitted_'):
            raise ValueError("Fit model first.")
        rng = np.random.RandomState(seed)
        n_trials = len(self.df)
        df_options = self.df['options'].values
        df_choice = self.df['choice'].values.astype(np.int64)
        sims = np.zeros((n_sims, n_trials), dtype=np.int64)

        probs_matrix = self.predict_proba(self.fitted_)
        for s in range(n_sims):
            for t in range(n_trials):
                opts = df_options[t]
                n_opts = len(opts)
                probs = probs_matrix[t, :n_opts]
                if np.isnan(probs).any() or probs.sum() <= 0:
                    probs = np.ones(n_opts) / n_opts
                sims[s, t] = rng.choice(opts, p=probs)
        sim_mean = sims.mean(axis=0)
        r2 = r2_score(df_choice.astype(float), sim_mean)
        return {'r2': r2, 'sims': sims, 'sim_mean': sim_mean}

    def parameter_recovery(self, template_df, fitted_params=None, n_subjects=20, seed=1, n_starts=20):
        """
        Генерация синтетических субъектов и оценка recovery.
        Изменения: более реалистичные диапазоны true-параметров и больше субъектов по умолчанию.
        """
        rng = np.random.RandomState(seed)
        trues, fits = [], []
        gen_model = HaffkeColdModel_Model3(template_df)

        # выбор realistic диапазонов для true-параметров (не крайние)
        for s in trange(n_subjects, desc="Parameter recovery"):
            if fitted_params is None:
                true = np.array([
                    rng.uniform(0.7, 1.4),   # alpha
                    rng.uniform(0.05, 1.2),  # delta
                    rng.uniform(0.3, 1.8),   # eta
                    rng.uniform(0.8, 8.0),   # theta
                    rng.uniform(0.0, 0.06)   # lapse
                ])
            else:
                # добавляем небольшую шумовую компонету к центр. значениям
                true = np.array(fitted_params) + rng.normal(scale=0.05, size=len(fitted_params))

            # сгенерировать выборы
            probs_matrix = gen_model.predict_proba(true)
            choices = []
            for t in range(len(template_df)):
                opts = template_df.loc[t, 'options']
                n_opts = len(opts)
                probs = probs_matrix[t, :n_opts]
                if np.isnan(probs).any() or probs.sum() <= 0:
                    probs = np.ones(n_opts) / n_opts
                choices.append(int(rng.choice(opts, p=probs)))

            sim_df = template_df.copy().reset_index(drop=True)
            sim_df['choice'] = choices

            # фитим модель по сгенерированным данным
            model_fit = HaffkeColdModel_Model3(sim_df)
            res = model_fit.fit(n_starts=n_starts, random_seed=seed + s, verbose=False)
            trues.append(true)
            fits.append(res.x)

        trues, fits = np.vstack(trues), np.vstack(fits)
        df_result = pd.DataFrame(
            np.hstack([trues, fits]),
            columns=[f"true_{p}" for p in ['alpha','delta','eta','theta','lapse']] +
                    [f"fit_{p}" for p in ['alpha','delta','eta','theta','lapse']]
        )
        corrs = {p: np.corrcoef(df_result[f"true_{p}"], df_result[f"fit_{p}"])[0,1]
                 for p in ['alpha','delta','eta','theta','lapse']}
        return {'df': df_result, 'corrs': corrs}
