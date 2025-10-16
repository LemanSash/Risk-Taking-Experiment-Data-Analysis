import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit  # sigmoid
from sklearn.metrics import r2_score
from tqdm import tqdm


class Par4Model:
    """
    Реализация Par4 (reparametrized 4-parameter model) из Park et al. (2021).

    Параметры модели:
        - phi (φ): prior belief of success (0 < φ < 1)
        - eta (η): updating coefficient (η > 0)
        - gamma (γ): risk-taking propensity (γ >= 0)
        - tau (τ): inverse temperature / consistency (τ >= 0)

    Основная логика:
        p_burst_k = 1 - (phi + eta * Σ successes) / (1 + eta * Σ pumps)
        nu_k      = -gamma / ln(1 - p_burst_k)
        p_pump(l) = sigmoid(tau * (nu_k - l))
    """

    def __init__(self, data_df=None, phi_init=0.1):
        self.data = None if data_df is None else data_df.copy()
        self.phi_init = phi_init

    # ---------- вспомогательная функция ----------
    @staticmethod
    def _compute_pburst(phi, eta, cum_successes, cum_pumps, eps=1e-12):
        num = phi + eta * cum_successes
        den = 1.0 + eta * cum_pumps
        val = 1.0 - (num / (den + eps))
        return float(np.clip(val, eps, 1.0 - eps))

    # ---------- log-likelihood ----------
    @staticmethod
    def nll(params, user_data, phi_init=0.1):
        phi, eta, gamma, tau = params

        if not (0.0 < phi < 1.0 and eta > 0 and gamma >= 0 and tau >= 0):
            return np.inf

        user_data = user_data.sort_values('trial_number')
        nll = 0.0
        eps = 1e-12
        cum_successes = 0.0
        cum_pumps = 0.0

        for _, row in user_data.iterrows():
            pumps = int(row['pumps'])
            popped = bool(row['popped'])

            # 1) вычисляем субъективную вероятность взрыва
            p_burst_k = Par4Model._compute_pburst(phi, eta, cum_successes, cum_pumps, eps=eps)

            # 2) оптимальное число насосов
            denom = np.log(max(eps, 1.0 - p_burst_k))
            nu_k = -gamma / denom

            # 3) log-likelihood (исправленный off-by-one)
            if popped:
                # Участник pump на всех шагах 1..pumps, последний вызвал взрыв
                for j in range(1, pumps + 1):
                    p_pump = expit(tau * (nu_k - j))
                    nll -= np.log(p_pump + eps)
            else:
                # Участник pump на 1..pumps, затем cash-out на j=pumps+1
                for j in range(1, pumps + 1):
                    p_pump = expit(tau * (nu_k - j))
                    nll -= np.log(p_pump + eps)
                p_next = expit(tau * (nu_k - (pumps + 1)))
                nll -= np.log(1.0 - p_next + eps)

            # 4) обновление кумулятивных счётчиков
            successes = pumps - 1 if popped else pumps
            cum_successes += successes
            cum_pumps += pumps

        return nll

    # ---------- подгонка ----------
    def fit(self, user_data, x0=None, bounds=None, verbose=False):
        if x0 is None:
            x0 = np.array([0.2, 0.1, 1.0, 1.0], dtype=float)
        if bounds is None:
            bounds = [(1e-3, 0.99), (1e-6, 50.0), (1e-6, 50.0), (1e-6, 50.0)]

        func = lambda p: Par4Model.nll(p, user_data, phi_init=self.phi_init)
        res = minimize(func, x0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 2000, 'ftol': 1e-8})

        if verbose:
            print("fit success:", res.success, "message:", res.message)
            print("params:", res.x)

        # fallback при неудаче
        if not res.success:
            alt_inits = [[0.1, 0.05, 0.5, 0.5], [0.5, 0.2, 2.0, 2.0], [0.05, 0.5, 1.0, 0.5]]
            for alt in alt_inits:
                ralt = minimize(func, np.array(alt), method='L-BFGS-B', bounds=bounds,
                                options={'maxiter': 2000, 'ftol': 1e-8})
                if ralt.success:
                    res = ralt
                    break
        return res.x

    # ---------- симуляция ----------
    def simulate(self, params, n_trials, max_pumps=64, explosion_sampler='uniform', seed=None):
        phi, eta, gamma, tau = params
        rng = np.random.RandomState(seed)
        data = []
        cum_successes = 0.0
        cum_pumps = 0.0

        for t in range(1, n_trials + 1):
            p_burst_k = Par4Model._compute_pburst(phi, eta, cum_successes, cum_pumps)
            denom = np.log(max(1e-12, 1.0 - p_burst_k))
            nu_k = -gamma / denom

            # скрытая точка взрыва
            explosion_point = rng.randint(1, max_pumps + 1)
            j = 0
            popped = False

            while True:
                j += 1
                p_pump = expit(tau * (nu_k - j))
                choose_pump = (rng.rand() < p_pump)
                if not choose_pump:
                    popped = False
                    break
                if j >= explosion_point:
                    popped = True
                    break
                if j >= max_pumps:
                    popped = False
                    break

            data.append({'trial_number': t, 'pumps': j, 'popped': int(popped), 'p_burst': p_burst_k})

            successes = j - 1 if popped else j
            cum_successes += successes
            cum_pumps += j

        return pd.DataFrame(data)

    # ---------- PPC ----------
    def predictive_check(self, real_data, verbose=True, seed=42):
        params_fit = self.fit(real_data)
        sim_df = self.simulate(params_fit, n_trials=len(real_data), seed=seed)
        merged = pd.merge(
            real_data[['trial_number', 'pumps']],
            sim_df[['trial_number', 'pumps']],
            on='trial_number', suffixes=('_real', '_sim')
        )
        r2 = r2_score(merged['pumps_real'], merged['pumps_sim'])
        if verbose:
            print("PPC Par4:")
            print(f"  phi={params_fit[0]:.4f}, eta={params_fit[1]:.4f}, gamma={params_fit[2]:.4f}, tau={params_fit[3]:.4f}")
            print(f"  Среднее pumps: реальное={merged['pumps_real'].mean():.2f}, симуляция={merged['pumps_sim'].mean():.2f}")
            print(f"  R²(real vs sim) = {r2:.3f}")
        return r2, params_fit, sim_df

    # ---------- recovery ----------
    def parameter_recovery(self, n_subjects=50, n_trials=200, seed_base=1000):
        rng = np.random.RandomState(12345)
        true_list, fit_list = [], []

        for i in tqdm(range(n_subjects), desc="Par4 parameter recovery"):
            phi_t = rng.uniform(0.05, 0.8)
            eta_t = rng.uniform(0.01, 1.0)
            gamma_t = rng.uniform(0.1, 5.0)
            tau_t = rng.uniform(0.1, 5.0)
            p_true = [phi_t, eta_t, gamma_t, tau_t]

            sim = self.simulate(p_true, n_trials=n_trials, seed=seed_base + i)
            try:
                p_fit = self.fit(sim)
            except Exception:
                p_fit = [np.nan] * 4

            true_list.append(p_true)
            fit_list.append(p_fit)

        cols = ['phi', 'eta', 'gamma', 'tau']
        df = pd.DataFrame(np.hstack([true_list, fit_list]),
                          columns=[f"true_{c}" for c in cols] + [f"fit_{c}" for c in cols])

        print("\nPar4 Parameter Recovery correlations:")
        for c in cols:
            r = np.corrcoef(df[f"true_{c}"], df[f"fit_{c}"])[0, 1]
            print(f"  {c}: r = {r:.3f}")
        return df
