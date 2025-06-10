import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
import pandas as pd
from sklearn.metrics import r2_score
from tqdm import tqdm

class STLDModel:
    """
    Модель Scaled Target Learning with Decay для BART.
    """
    def __init__(self, data_df):
        self.data = data_df.copy()

    @staticmethod
    def nll(params, user_data):
        w1, vwin, vloss, alpha, beta = params
        user_data = user_data.sort_values('trial_number')

        w = w1
        lambda_t = 1.0
        nll = 0.0
        eps = 1e-10

        for _, row in user_data.iterrows():
            pumps  = int(row['pumps'])
            popped = bool(row['popped'])

            # 1) likelihood
            for k in range(1, pumps+1):
                p_pump = expit(beta * (w - k))
                if k < pumps:
                    nll -= np.log(p_pump + eps)
                else:
                    if popped:
                        nll -= np.log(p_pump + eps)
                    else:
                        nll -= np.log(1 - p_pump + eps)
                    break

            # 2) update w один раз с линейным decay
            eta = max(0.0, lambda_t)
            if popped:
                delta = (pumps - 1) - w
                w += eta * (vloss * delta)
            else:
                delta = pumps - w
                w += eta * (vwin * delta)

            # 3) update lambda_t линейно
            lambda_t = max(0.0, lambda_t - alpha)
        return nll

    def fit(self, user_data, x0=None, bounds=None):
        """
        Подгонка параметров модели STL-D для одного пользователя.
        """
        # Начальные приближения и границы
        init_params = [10.0, 0.3, 0.7, 0.2, 2.0]
        bounds = [
            (1.0, 50.0),    # w1
            (0.0, 1.0),     # vwin
            (0.0, 1.0),     # vloss
            (0.01, 1.0),    # alpha
            (0.01, 20.0)    # beta
        ]

        result = minimize(
            self.nll,
            init_params,
            args=(user_data),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-6}
        )
        return result.x

    def simulate(self, params, n_trials, max_pumps=64, seed=None):
        rng = np.random.RandomState(seed)
        w1, vwin, vloss, alpha, beta = params

        w = w1
        lambda_t = 1.0
        data = []
        
        for t in range(1, n_trials+1):
            kappa = rng.randint(1, max_pumps+1)
            k = 0
            popped = False
            
            while True:
                k += 1
                p_pump = expit(beta * (w - k))
                if rng.rand() > p_pump:
                    popped = False
                    break
                if k == kappa:
                    popped = True
                    break
            
            data.append({'trial_number': t, 'pumps': k, 'popped': popped})

            eta = max(0.0, lambda_t)
            if popped:
                delta = (k - 1) - w
                w += eta * (vloss * delta)
            else:
                delta = k - w
                w += eta * (vwin  * delta)
            lambda_t = max(0.0, lambda_t - alpha)
        sim_df = pd.DataFrame(data)
        return sim_df
    
    def predictive_check(self, real_data):
        """
        1) Подгоняем модель к real_data → получаем params_fit.
        2) Симулируем поведение той же длины.
        3) Сравниваем среднее число насосов real vs sim.
        """
        # Подгонка
        params_fit = self.fit(real_data)
        
        # Симуляция
        sim_df = self.simulate(params_fit, n_trials=len(real_data), seed=42)
        
        # Сравнение средних pumps
        real_mean = real_data['pumps'].mean()
        sim_mean  = sim_df['pumps'].mean()
        
        # Считаем R² по вектору насосов (real vs sim)
        # Для этого упорядочим по trial_number
        merged = pd.merge(
            real_data[['trial_number','pumps']],
            sim_df [['trial_number','pumps']],
            on='trial_number',
            suffixes=('_real','_sim')
        )
        r2 = r2_score(merged['pumps_real'], merged['pumps_sim'])
        
        print("PPC STL-D:")
        print(f"  Среднее pumps реальное = {real_mean:.2f}, симуляция = {sim_mean:.2f}")
        print(f"  R² между реальными и симулированными pumps: {r2:.3f}")
        return r2
    
    def parameter_recovery(self, n_subjects=50, n_trials=200):
        """
        Parameter Recovery для STL-D (Zhou et al., 2021), после правок модели:
        1) Генерируем true_params uniform в их bounds.
        2) Симулируем по исправленному simulate_stld (с единственным κ и линейным decay).
        3) Подгоняем модель fit_stld_model(sim).
        4) Считаем корреляции true vs fit.
        """
        true_list = []
        fit_list  = []
        
        for i in tqdm(range(n_subjects), desc="Parameter recovery STL-D"):
            # 1) True-параметры из uniform в их bounds
            w1    = np.random.uniform(1.0, 50.0)
            vwin  = np.random.rand()
            vloss = np.random.rand()
            alpha = np.random.uniform(0.01, 1.0)
            beta  = np.random.uniform(0.01, 20.0)
            p_true = [w1, vwin, vloss, alpha, beta]
            
            # 2) Симуляция поведения
            sim = self.simulate(p_true, n_trials, max_pumps=64, seed=100+i)
            
            # 3) Подгонка модели к сим-данным
            p_fit = self.fit(sim)  # обновлённый fit без l2_penalty
            
            true_list.append(p_true)
            fit_list.append(p_fit)
        
        # 4) Считаем корреляции true vs fit
        cols = ['w1','vwin','vloss','alpha','beta']
        true_arr = np.array(true_list)
        fit_arr  = np.array(fit_list)
        
        df = pd.DataFrame(
            np.hstack([true_arr, fit_arr]),
            columns=[f"true_{c}" for c in cols] + [f"fit_{c}" for c in cols]
        )
        
        print("Parameter Recovery STL-D: корреляции (true vs fit)")
        for c in cols:
            r = np.corrcoef(df[f"true_{c}"], df[f"fit_{c}"])[0,1]
            print(f"  {c}: r = {r:.3f}")
        return df
