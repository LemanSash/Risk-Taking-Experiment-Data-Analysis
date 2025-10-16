import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit  # sigmoid
from sklearn.metrics import r2_score
from tqdm import tqdm

class EWMVModel:
    """
    Exponential-Weight Mean-Variance (EWMV) model (Park et al., 2021).

    Параметры в векторе params: [psi, xi, rho, tau, lam]
      - psi (ψ): prior belief of burst, 0 < psi < 1
      - xi  (ξ): updating exponent, ξ >= 0
      - rho (ρ): mean-variance coefficient (risk preference)
      - tau (τ): inverse temperature / consistency, τ >= 0
      - lam (λ): loss aversion, λ >= 0

    Формулы:
      p_burst_k = exp(-ξ * N_{k-1}) * ψ + (1 - exp(-ξ * N_{k-1})) * (S_{k-1} / N_{k-1})
      U_pump^{kl} = (1 - p_burst_k) * r - p_burst_k * λ * (l-1) * r
                     + ρ * p_burst_k * (1 - p_burst_k) * (r + λ * (l-1) * r)**2
      p_pump = sigmoid( τ * U_pump^{kl} )

    Реализация использует r=1 по умолчанию.
    Источник: Park et al., Journal of Mathematical Psychology (2021). :contentReference[oaicite:6]{index=6}
    """

    def __init__(self, data_df=None, r=1.0, psi_init=0.1):
        self.data = None if data_df is None else data_df.copy()
        self.r = float(r)
        self.psi_init = psi_init

    @staticmethod
    def _safe_div(a, b, eps=1e-12):
        return a / (b + eps)

    @staticmethod
    def _compute_pburst(psi, xi, cum_successes, cum_pumps, eps=1e-12):
        """
        p_burst_k = exp(-xi * N) * psi + (1 - exp(-xi * N)) * (S / N)
        Если N == 0 -> P_emp = 0.
        """
        weight = np.exp(- xi * cum_pumps)
        if cum_pumps <= 0:
            P_emp = 0.0
        else:
            #P_emp = cum_successes / (cum_pumps + eps)
            P_emp = (cum_pumps - cum_successes) / (cum_pumps + eps)
        val = weight * psi + (1.0 - weight) * P_emp
        return float(np.clip(val, eps, 1.0 - eps))

    @staticmethod
    def _utility_pump(p_burst_k, l, r, rho, lam):
        """
        U_pump^{kl} по Eq.(16) Park et al. (EWMV):
          U = (1 - p) * r - p * lam * (l-1) * r
              + rho * p * (1 - p) * ( r + lam * (l-1) * r )^2
        """
        gain_term = (1.0 - p_burst_k) * r
        loss_amt = (l - 1) * r
        loss_term = p_burst_k * lam * loss_amt
        variance_term = rho * p_burst_k * (1.0 - p_burst_k) * (r + lam * loss_amt) ** 2
        return gain_term - loss_term + variance_term

    
    @staticmethod
    def nll(params, user_data, r=1.0):
        """
        Negative log-likelihood (per-subject).
        user_data must contain columns: ['trial_number','pumps','popped'].
        """
        psi, xi, rho, tau, lam = params

        if not (0.0 < psi < 1.0 and xi >= 0 and tau >= 0 and lam >= 0):
            return 1e9

        user_data = user_data.sort_values('trial_number')
        eps = 1e-12
        nll = 0.0
        cum_successes = 0.0
        cum_pumps = 0.0

        for _, row in user_data.iterrows():
            pumps = int(row['pumps'])
            popped = bool(row['popped'])

            # 1) belief перед trial
            p_burst_k = EWMVModel._compute_pburst(psi, xi, cum_successes, cum_pumps, eps=eps)

            # 2) цикл по pump-возможностям
            for j in range(1, pumps + 1):
                U_pump = EWMVModel._utility_pump(p_burst_k, j, r, rho, lam)
                p_pump = expit(tau * U_pump)
                if j < pumps:
                    nll -= np.log(p_pump + eps)
                else:
                    if popped:
                        # взрыв на j-м pump
                        nll -= np.log(p_pump + eps)
                    else:
                        # отказ pumpнуть на следующем шаге (j+1)
                        U_next = EWMVModel._utility_pump(p_burst_k, j + 1, r, rho, lam)
                        p_next = expit(tau * U_next)
                        nll -= np.log(1.0 - p_next + eps)
                    break

            # 3) обновляем кумулятивные счётчики
            successes = (pumps - 1) if popped else pumps
            cum_successes += successes
            cum_pumps += pumps

        return nll


    def fit(self, user_data, x0=None, bounds=None, verbose=False):
        """
        MLE-fit для одного пользователя.
        Рекомендуемые x0 и bounds:
          x0 = [0.1, 0.05, 0.1, 1.0, 1.0]
          bounds = [(1e-3, 0.99), (0.0, 50.0), (-5.0, 5.0), (1e-6, 50.0), (0.0, 50.0)]
        Возвращает вектор параметров [psi, xi, rho, tau, lam].
        """
        if x0 is None:
            x0 = np.array([0.1, 0.05, 0.1, 1.0, 1.0], dtype=float)
        if bounds is None:
            bounds = [(1e-3, 0.99), (0.0, 50.0), (-5.0, 5.0), (1e-6, 50.0), (0.0, 50.0)]

        func = lambda p: EWMVModel.nll(p, user_data, r=self.r)
        res = minimize(func, x0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 2000, 'ftol': 1e-8})
        if verbose:
            print("fit success:", res.success, "message:", res.message)
            print("params:", res.x)

        if not res.success or res.fun > 1e6:
            alt_inits = [
                np.random.uniform(low=[0.01, 0.0, -1, 0.1, 0.1], high=[0.8, 1, 1, 5, 5])
                for _ in range(10)
            ]
            for alt in alt_inits:
                ralt = minimize(func, alt, method='L-BFGS-B', bounds=bounds,
                                options={'maxiter':2000, 'ftol':1e-8})
                if ralt.success and ralt.fun < res.fun:
                    res = ralt

        return res.x

    def simulate(self, params, n_trials, max_pumps=64, seed=None):
        psi, xi, rho, tau, lam = params
        rng = np.random.RandomState(seed)
        data = []
        cum_successes = 0.0
        cum_pumps = 0.0

        for t in range(1, n_trials + 1):
            p_burst_k = EWMVModel._compute_pburst(psi, xi, cum_successes, cum_pumps)
            explosion_point = rng.randint(1, max_pumps + 1)

            pumps = 0
            popped = False
            while True:
                pumps += 1
                U_pump = EWMVModel._utility_pump(p_burst_k, pumps, self.r, rho, lam)
                p_pump = expit(tau * U_pump)
                if rng.rand() > p_pump:
                    # отказ pumpнуть на этом шаге → cash
                    popped = False
                    pumps -= 1  # реально выполнено накачек меньше на одну
                    break
                if pumps >= explosion_point:
                    popped = True
                    break
                if pumps >= max_pumps:
                    popped = False
                    break

            data.append({'trial_number': t, 'pumps': pumps, 'popped': popped, 'p_burst': p_burst_k})
            successes = (pumps - 1) if popped else pumps
            cum_successes += successes
            cum_pumps += pumps

        return pd.DataFrame(data)

    def predictive_check(self, real_data, verbose=True):
        """
        Подгонка и единичная симуляция для PPC (как в твоих других моделях).
        Возвращает (r2, params_fit, sim_df).
        """
        params_fit = self.fit(real_data)
        sim_df = self.simulate(params_fit, n_trials=len(real_data), seed=42)
        real_mean = real_data['pumps'].mean()
        sim_mean = sim_df['pumps'].mean()
        merged = pd.merge(real_data[['trial_number','pumps']], sim_df[['trial_number','pumps']],
                          on='trial_number', suffixes=('_real','_sim'))
        r2 = r2_score(merged['pumps_real'], merged['pumps_sim'])
        if verbose:
            print("PPC EWMV:")
            print(f" params_fit: psi={params_fit[0]:.4f}, xi={params_fit[1]:.4f}, rho={params_fit[2]:.4f}, tau={params_fit[3]:.4f}, lam={params_fit[4]:.4f}")
            print(f" mean pumps real = {real_mean:.2f}, sim = {sim_mean:.2f}, R^2 = {r2:.3f}")
        return r2, params_fit, sim_df

    def parameter_recovery(self, n_subjects=50, n_trials=200, seed_base=1000):
        """
        Простейшая процедура recovery: симулируем n_subjects с генерацией параметров
        в рабочем диапазоне и затем оцениваем (MLE).
        Возвращает DataFrame с true и fit параметрами и печатает корреляции.
        """
        rng = np.random.RandomState(12345)
        true_list = []
        fit_list = []

        for i in tqdm(range(n_subjects), desc="EWMV parameter recovery"):
            psi_t = rng.uniform(0.05, 0.8)
            xi_t  = rng.uniform(0.0, 1.0)
            rho_t = rng.uniform(-1.0, 1.0)
            tau_t = rng.uniform(0.1, 5.0)
            lam_t = rng.uniform(0.1, 5.0)
            p_true = [psi_t, xi_t, rho_t, tau_t, lam_t]

            sim = self.simulate(p_true, n_trials=n_trials, seed=seed_base + i)
            try:
                p_fit = self.fit(sim)
            except Exception as e:
                p_fit = [np.nan]*5

            true_list.append(p_true)
            fit_list.append(p_fit)

        cols = ['psi','xi','rho','tau','lam']
        df = pd.DataFrame(np.hstack([np.array(true_list), np.array(fit_list)]),
                          columns=[f"true_{c}" for c in cols] + [f"fit_{c}" for c in cols])
        print("EWMV parameter recovery correlations (true vs fit):")
        for c in cols:
            r = np.corrcoef(df[f"true_{c}"].astype(float), df[f"fit_{c}"].astype(float))[0,1]
            print(f"  {c}: r = {r:.3f}")
        return df
