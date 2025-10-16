import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit  # sigmoid
from sklearn.metrics import r2_score
from tqdm import tqdm

class STLModel:
    """
    Scaled Target Learning (STL) model — версия без decay, точно по Zhou et al. (2021).

    Формулы:
      Вероятность накачки:
        p_pump = 1 / (1 + exp(beta * (l - w_t)))
      Обновление w_t (мультипликативное):
        if popped:
            w_{t+1} = w_t * (1 - vloss * (1 - pumps / max_pumps))
        else:
            w_{t+1} = w_t * (1 + vwin * (pumps / max_pumps))
    Параметры: [w1, vwin, vloss, beta]
    """
    
    def __init__(self, data_df=None, max_pumps=64):
        self.data = None if data_df is None else data_df.copy()
        self.max_pumps = int(max_pumps)
    
    @staticmethod
    def nll(params, user_data, max_pumps=64):
        w1, vwin, vloss, beta = params
        if not (1.0 <= w1 <= max_pumps and 0.0 <= vwin <= 1.0 and 0.0 <= vloss <= 1.0 and beta > 0):
            return 1e9
        
        user_data = user_data.sort_values('trial_number').reset_index(drop=True)
        eps = 1e-12
        nll = 0.0
        w = float(w1)
        
        for _, row in user_data.iterrows():
            pumps = int(row['pumps'])
            popped = bool(row['popped'])
            
            for k in range(1, pumps + 1):
                p_pump = 1.0 / (1.0 + np.exp(beta * (k - w)))  # формула статьи
                if k < pumps:
                    nll -= np.log(p_pump + eps)
                else:
                    if popped:
                        nll -= np.log(p_pump + eps)
                    else:
                        nll -= np.log(1.0 - p_pump + eps)
                    break
            
            # Мультипликативное обновление w
            if popped:
                w = w * (1.0 - vloss * (1.0 - pumps / max_pumps))
            else:
                w = w * (1.0 + vwin * (pumps / max_pumps))
            
            # Ограничение w
            w = float(np.clip(w, 1.0, max_pumps))
        
        return nll
    
    def fit(self, user_data, x0=None, bounds=None, verbose=False):
        if x0 is None:
            x0 = np.array([10.0, 0.3, 0.7, 2.0], dtype=float)
        if bounds is None:
            bounds = [
                (1.0, float(self.max_pumps)),  # w1
                (0.0, 1.0),                    # vwin
                (0.0, 1.0),                    # vloss
                (1e-6, 50.0)                   # beta
            ]
        func = lambda p: STLModel.nll(p, user_data, max_pumps=self.max_pumps)
        res = minimize(func, x0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 2000, 'ftol': 1e-8})
        if verbose:
            print("STL fit success:", res.success, res.message)
        if not res.success:
            restarts = [[5.0,0.2,0.2,1.0],[15.0,0.5,0.5,3.0],[8.0,0.1,0.9,0.5]]
            for alt in restarts:
                ralt = minimize(func, np.array(alt), method='L-BFGS-B', bounds=bounds,
                                options={'maxiter':2000, 'ftol':1e-8})
                if ralt.success:
                    res = ralt
                    break
        return res.x
    
    def simulate(self, params, n_trials, max_pumps=None, seed=None):
        if max_pumps is None:
            max_pumps = self.max_pumps
        rng = np.random.RandomState(seed)
        w1, vwin, vloss, beta = params
        w = float(w1)
        data = []
        
        for t in range(1, int(n_trials) + 1):
            explosion_point = rng.randint(1, int(max_pumps) + 1)
            k = 0
            popped = False
            while True:
                k += 1
                p_pump = 1.0 / (1.0 + np.exp(beta * (k - w)))
                if rng.rand() > p_pump:
                    popped = False
                    break
                if k >= explosion_point:
                    popped = True
                    break
                if k >= max_pumps:
                    popped = False
                    break
            data.append({'trial_number': t, 'pumps': int(k), 'popped': bool(popped), 'w_before': w})
            
            # Мультипликативное обновление w
            if popped:
                w = w * (1.0 - vloss * (1.0 - k / max_pumps))
            else:
                w = w * (1.0 + vwin * (k / max_pumps))
            w = float(np.clip(w, 1.0, max_pumps))
        
        return pd.DataFrame(data)
    
    def predictive_check(self, real_data, verbose=True):
        params_fit = self.fit(real_data)
        sim_df = self.simulate(params_fit, n_trials=len(real_data), seed=42)
        merged = pd.merge(
            real_data[['trial_number','pumps']],
            sim_df [['trial_number','pumps']],
            on='trial_number',
            suffixes=('_real','_sim')
        )
        r2 = r2_score(merged['pumps_real'], merged['pumps_sim'])
        if verbose:
            print("PPC STL (по статье):")
            print(f"  params_fit: w1={params_fit[0]:.3f}, vwin={params_fit[1]:.3f}, "
                  f"vloss={params_fit[2]:.3f}, beta={params_fit[3]:.3f}")
            print(f"  R^2 = {r2:.3f}")
        return r2, params_fit, sim_df
    
    def parameter_recovery(self, n_subjects=50, n_trials=200, seed_base=100):
        rng = np.random.RandomState(12345)
        true_list = []
        fit_list = []
        for i in tqdm(range(int(n_subjects)), desc="STL parameter recovery"):
            w1 = rng.uniform(1.0, min(30.0, self.max_pumps))
            vwin = rng.uniform(0.0, 1.0)
            vloss = rng.uniform(0.0, 1.0)
            beta = rng.uniform(0.1, 5.0)
            p_true = [w1, vwin, vloss, beta]
            
            sim = self.simulate(p_true, n_trials=int(n_trials), seed=seed_base + i)
            try:
                p_fit = self.fit(sim)
            except Exception:
                p_fit = [np.nan]*4
            true_list.append(p_true)
            fit_list.append(p_fit)
        
        cols = ['w1','vwin','vloss','beta']
        df = pd.DataFrame(np.hstack([np.array(true_list), np.array(fit_list)]),
                          columns=[f"true_{c}" for c in cols] + [f"fit_{c}" for c in cols])
        print("STL (по статье) parameter recovery correlations (true vs fit):")
        for c in cols:
            r = np.corrcoef(df[f"true_{c}"].astype(float), df[f"fit_{c}"].astype(float))[0,1]
            print(f"  {c}: r = {r:.3f}")
        return df

