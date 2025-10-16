import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit  # sigmoid
from sklearn.metrics import r2_score
from tqdm import tqdm

class EWModel:
    """
    Exponential-Weight (EW) model from Park et al. (2021).

    Параметры в векторе params: [psi, xi, rho, tau, lam]
      - psi (ψ): prior belief of burst, 0 < psi < 1
      - xi  (ξ): updating exponent (ξ >= 0)
      - rho (ρ): risk preference (can be positive or negative; practical bounds used)
      - tau (τ): inverse temperature / consistency (τ >= 0)
      - lam (λ): loss aversion (λ >= 0)

    Формулы:
      - p_burst_k = exp(-ξ * cum_pumps) * ψ + (1 - exp(-ξ * cum_pumps)) * P_{k-1}
        где P_{k-1} = cum_successes / cum_pumps  (обсервабельная доля успешных pump'ов до trial k).
      - U_pump^{kl} = (1 - p_burst_k) * r^ρ - p_burst_k * λ * ((l-1)*r)^ρ   (обычно r=1)
      - p_pump = sigmoid( τ * U_pump^{kl} )
    Источник: Park et al., J. Math. Psych. 2021 (Sec. 2.3.4–2.3.6). :contentReference[oaicite:5]{index=5}
    """

    def __init__(self, data_df=None, r=1.0, psi_init=0.1):
        """
        data_df: (optional) global DataFrame (not обязательно).
        r: reward per successful pump (по умолчанию 1).
        psi_init: значение prior для p_burst перед первым trial (если not provided in rows).
        """
        self.data = None if data_df is None else data_df.copy()
        self.r = float(r)
        self.psi_init = psi_init

    @staticmethod
    def _safe_div(a, b, eps=1e-12):
        return a / (b + eps)

    @staticmethod
    def _compute_pburst(psi, xi, cum_successes, cum_pumps, eps=1e-12):
        """
        p_burst_k = exp(-xi * cum_pumps) * psi + (1 - exp(-xi * cum_pumps)) * P_{k-1}
        где P_{k-1} = cum_successes / cum_pumps (если cum_pumps==0, используем 0.0).
        """
        weight = np.exp(- xi * cum_pumps)
        if cum_pumps <= 0:
            P_emp = 0.0
        else:
            # observed explosion probability = exploded / total_pumps
            P_emp = (cum_pumps - cum_successes) / (cum_pumps + eps)
        val = weight * psi + (1.0 - weight) * P_emp
        # clip в (eps,1-eps)
        return float(np.clip(val, eps, 1.0 - eps))

    @staticmethod
    def _utility_pump(p_burst_k, l, r, rho, lam):
        """
        U_pump = (1 - p_burst) * r^rho - p_burst * lam * ((l-1)*r)^rho
        Примечание: если rho==0, r^rho == 1 (логика power utility).
        """
        # чтобы поддерживать rho==0 корректно используем np.power
        gain_term = (1.0 - p_burst_k) * (r ** rho)
        loss_amount = ((l - 1) * r)
        # если loss_amount == 0 и rho < 1, 0**rho -> 0, OK
        loss_term = p_burst_k * lam * (loss_amount ** rho)
        return gain_term - loss_term

    @staticmethod
    def nll(params, user_data, r=1.0, psi_init=0.1):
        """
        Negative log-likelihood для одного пользователя.
        params = [psi, xi, rho, tau, lam]
        user_data: DataFrame с колонками ['trial_number','pumps','popped'] (popped True/False).
        """
        psi, xi, rho, tau, lam = params
        # ограничения параметров (простая защита; при желании можно более гибко)
        if not (0.0 < psi < 1.0 and xi >= 0 and tau >= 0 and lam >= 0):
            return 1e9
        # rho допускаем любой знак, но ограничим по числу в оптимизации bounds

        user_data = user_data.sort_values('trial_number')
        nll = 0.0
        eps = 1e-12

        cum_successes = 0.0  # суммарное число успешных насосов (не включая exploded last pump)
        cum_pumps = 0.0      # суммарное число попыток pump (включая те, что привели к explosion)

        for _, row in user_data.iterrows():
            pumps = int(row['pumps'])
            popped = bool(row['popped'])

            p_burst_k = EWModel._compute_pburst(psi, xi, cum_successes, cum_pumps, eps=eps)
            # likelihood: просматриваем j = 1..pumps
            for j in range(1, pumps + 1):
                U_pump = EWModel._utility_pump(p_burst_k, j, r, rho, lam)
                p_pump = expit(tau * U_pump)  # sigmoid
                if j < pumps:
                    # промежуточные выборы: делал pump
                    nll -= np.log(p_pump + eps)
                else:
                    # последний шаг: либо сделал pump (и popped), либо решил cash (did not pump on j)
                    if popped:
                        nll -= np.log(p_pump + eps)
                    else:
                        nll -= np.log(1.0 - p_pump + eps)
                    break

            # обновляем кумулятивные счётчики для следующего trial:
            # successful pumps on this trial = pumps if no pop, else pumps-1
            successes = (pumps - 1) if popped else pumps
            cum_successes += successes
            cum_pumps += pumps

        return nll

    def fit(self, user_data, x0=None, bounds=None, verbose=False):
        """
        MLE-подгонка: минимизация nll.
        Рекомендованные начальные значения и границы:
           psi in (1e-3, 0.99), xi in (0, 10), rho in (-2, 2), tau in (1e-4, 50), lam in (0, 20)
        """
        if x0 is None:
            x0 = np.array([0.1, 0.05, 0.5, 1.0, 1.0], dtype=float)
        if bounds is None:
            #bounds = [(1e-3, 0.99), (0.0, 50.0), (-2.0, 2.0), (1e-6, 50.0), (0.0, 50.0)]
            # bounds default:
            bounds = [(1e-3, 0.99), (0.0, 50.0), (1e-6, 2.0), (1e-6, 50.0), (0.0, 50.0)]

        func = lambda p: EWModel.nll(p, user_data, r=self.r, psi_init=self.psi_init)
        res = minimize(func, x0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter':2000, 'ftol':1e-8})
        if verbose:
            print("fit success:", res.success, "message:", res.message)
        # попытки рестартов при неудаче
        if not res.success:
            for alt in [[0.2,0.1,0.5,1.0,1.0],[0.05,0.01,1.0,0.5,0.5],[0.2,0.5,-0.5,2.0,1.0]]:
                ratt = minimize(func, np.array(alt), method='L-BFGS-B', bounds=bounds,
                                 options={'maxiter':2000, 'ftol':1e-8})
                if ratt.success:
                    res = ratt
                    break
        return res.x

    def simulate(self, params, n_trials, max_pumps=64, seed=None):
        """
        Симуляция агента c params = [psi, xi, rho, tau, lam].
        Возвращает DataFrame ['trial_number','pumps','popped','p_burst'].
        """
        psi, xi, rho, tau, lam = params
        rng = np.random.RandomState(seed)
        data = []
        cum_successes = 0.0
        cum_pumps = 0.0

        for t in range(1, n_trials+1):
            p_burst_k = EWModel._compute_pburst(psi, xi, cum_successes, cum_pumps)
            # генерируем скрытую точку взрыва (uniform 1..max_pumps)
            explosion_point = rng.randint(1, max_pumps + 1)

            j = 0; popped = False
            while True:
                j += 1
                U_pump = EWModel._utility_pump(p_burst_k, j, self.r, rho, lam)
                p_pump = expit(tau * U_pump)
                if rng.rand() > p_pump:
                    popped = False; break
                # participant chose pump; check explosion
                if j >= explosion_point:
                    popped = True; break
                if j >= max_pumps:
                    popped = False; break

            data.append({'trial_number': t, 'pumps': j, 'popped': bool(popped), 'p_burst': p_burst_k})

            successes = (j - 1) if popped else j
            cum_successes += successes
            cum_pumps += j

        return pd.DataFrame(data)

    def predictive_check(self, real_data, verbose=True):
        params_fit = self.fit(real_data)
        sim_df = self.simulate(params_fit, n_trials=len(real_data), seed=42)
        real_mean = real_data['pumps'].mean()
        sim_mean = sim_df['pumps'].mean()
        merged = pd.merge(real_data[['trial_number','pumps']], sim_df[['trial_number','pumps']],
                          on='trial_number', suffixes=('_real','_sim'))
        r2 = r2_score(merged['pumps_real'], merged['pumps_sim'])
        if verbose:
            print("PPC EW model:")
            print(f" params_fit: psi={params_fit[0]:.4f}, xi={params_fit[1]:.4f}, rho={params_fit[2]:.4f}, tau={params_fit[3]:.4f}, lam={params_fit[4]:.4f}")
            print(f" mean pumps real={real_mean:.2f}, sim={sim_mean:.2f}, R^2={r2:.3f}")
        return r2, params_fit, sim_df

    def parameter_recovery(self, n_subjects=50, n_trials=200, seed_base=1000):
        rng = np.random.RandomState(12345)
        true_list = []
        fit_list = []
        for i in tqdm(range(n_subjects), desc="EW parameter recovery"):
            psi_t = rng.uniform(0.05, 0.8)
            xi_t = rng.uniform(0.0, 1.0)
            rho_t = rng.uniform(0.1, 1.9)
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
        print("EW parameter recovery correlations (true vs fit):")
        for c in cols:
            r = np.corrcoef(df[f"true_{c}"].astype(float), df[f"fit_{c}"].astype(float))[0,1]
            print(f"  {c}: r = {r:.3f}")
        return df
