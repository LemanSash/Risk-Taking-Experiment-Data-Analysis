import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import r2_score
from tqdm import tqdm

class STLDModel:
    """
    STL-Decay модель полностью соответствующая Zhou et al., 2021.
    Для одного типа шаров с фиксированным nmax = 64.
    """
    def __init__(self, data_df, nmax=64):
        """
        data_df: DataFrame с BART данными
        nmax: максимальное количество накачек для всех шаров
        """
        self.data = data_df.copy()
        self.nmax = nmax

    @staticmethod
    def nll(params, user_data, nmax=64):
        w1, vwin, vloss, a, beta = params
        user_data = user_data.sort_values('trial_number')

        w = w1
        nll = 0.0
        eps = 1e-10

        for idx, row in user_data.iterrows():
            pumps = int(row['pumps'])
            popped = bool(row['popped'])

            # likelihood по каждому шагу накачки
            for k in range(1, pumps + 1):
                p_pump = 1.0 / (1.0 + np.exp(beta * (k - w)))
                if k < pumps:
                    nll -= np.log(p_pump + eps)
                else:
                    if popped:
                        nll -= np.log(p_pump + eps)
                    else:
                        nll -= np.log(1 - p_pump + eps)
                    break

            # обновление целевого уровня w
            if popped:
                w *= 1 - (vloss * (1 - pumps / nmax)) / (1 + a * idx)
            else:
                w *= 1 + (vwin * pumps / nmax) / (1 + a * idx)

        return nll

    def fit(self, user_data, x0=None, bounds=None):
        init_params = [10.0, 0.3, 0.7, 0.1, 2.0]  # w1, vwin, vloss, a, beta
        bounds = [
            (1.0, 50.0),    # w1
            (0.0, 1.0),     # vwin
            (0.0, 1.0),     # vloss
            (0.0, 1.0),     # a
            (0.01, 10.0)    # beta
        ]

        result = minimize(
            self.nll,
            init_params,
            args=(user_data, self.nmax),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-6}
        )
        return result.x

    def simulate(self, params, n_trials, seed=None):
        rng = np.random.RandomState(seed)
        w1, vwin, vloss, a, beta = params
        w = w1
        data = []

        for t in range(n_trials):
            # случайный "порог" взрыва
            kappa = rng.randint(1, self.nmax + 1)
            k = 0
            popped = False

            while True:
                k += 1
                z = np.clip(beta * (k - w), -700, 700)
                p_pump = 1.0 / (1.0 + np.exp(z))
                #p_pump = 1.0 / (1.0 + np.exp(beta * (k - w)))
                if rng.rand() > p_pump:
                    popped = False
                    break
                if k == kappa:
                    popped = True
                    break

            data.append({'trial_number': t+1, 'pumps': k, 'popped': popped})

            # обновление w
            if popped:
                w *= 1 - (vloss * (1 - k / self.nmax)) / (1 + a * t)
            else:
                w *= 1 + (vwin * k / self.nmax) / (1 + a * t)

        return pd.DataFrame(data)

    def predictive_check(self, real_data):
        params_fit = self.fit(real_data)
        sim_df = self.simulate(params_fit, n_trials=len(real_data), seed=42)

        real_mean = real_data['pumps'].mean()
        sim_mean = sim_df['pumps'].mean()

        merged = pd.merge(
            real_data[['trial_number','pumps']],
            sim_df[['trial_number','pumps']],
            on='trial_number',
            suffixes=('_real','_sim')
        )
        r2 = r2_score(merged['pumps_real'], merged['pumps_sim'])

        print("PPC STL-D (Article, nmax=64):")
        print(f"  Среднее pumps реальное = {real_mean:.2f}, симуляция = {sim_mean:.2f}")
        print(f"  R² между реальными и симулированными pumps: {r2:.3f}")
        return r2

    def parameter_recovery(self, n_subjects=50, n_trials=200):
        true_list = []
        fit_list = []

        for i in tqdm(range(n_subjects), desc="Parameter recovery STL-D (Article)"):
            w1 = np.random.uniform(1.0, 50.0)
            vwin = np.random.rand()
            vloss = np.random.rand()
            a = np.random.uniform(0.0, 1.0)
            beta = np.random.uniform(0.01, 10.0)
            p_true = [w1, vwin, vloss, a, beta]

            sim = self.simulate(p_true, n_trials, seed=100+i)
            p_fit = self.fit(sim)

            true_list.append(p_true)
            fit_list.append(p_fit)

        cols = ['w1','vwin','vloss','a','beta']
        df = pd.DataFrame(
            np.hstack([np.array(true_list), np.array(fit_list)]),
            columns=[f"true_{c}" for c in cols] + [f"fit_{c}" for c in cols]
        )

        print("Parameter Recovery STL-D (Article, nmax=64): корреляции (true vs fit)")
        for c in cols:
            r = np.corrcoef(df[f"true_{c}"], df[f"fit_{c}"])[0,1]
            print(f"  {c}: r = {r:.3f}")
        return df
