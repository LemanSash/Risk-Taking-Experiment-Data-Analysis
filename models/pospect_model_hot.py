import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import r2_score
import pandas as pd
from tqdm import tqdm

class HOTModel:
    """
    Модель Value plus Sequential Exploration для IGT.
    """
    def __init__(self, data_df):
        self.data = data_df.copy()

    @staticmethod
    def utility(x, rho, lam):
        """
        Prospect‐theory utility без вычисления x**rho при x<0.
        """
        x = np.asarray(x)
        out = np.empty_like(x, dtype=float)
        pos_mask = x >= 0
        neg_mask = ~pos_mask
        # только там, где x>=0, выполняем x**rho
        out[pos_mask] = np.power(x[pos_mask], rho)
        # только там, где x<0, выполняем abs(x)**rho
        out[neg_mask] = -lam * np.power(np.abs(x[neg_mask]), rho)
        return out
    
    def nll(self, params, flips_df):
        rho, lam, beta = params
        nll = 0.0
        eps = 1e-9

        for trial, trial_df in flips_df.groupby('trial_number', sort=False):
            gain_amt = trial_df['gain_amount'].iloc[0]
            loss_amt = trial_df['loss_amount'].iloc[0]
            popped   = trial_df['popped'].iloc[0]  # ← ключевое изменение

            # утилиты
            u_gain = self.utility(gain_amt, rho, lam)
            u_loss = self.utility(-loss_amt, rho, lam)

            EU_raw = 0.5 * u_gain + 0.5 * u_loss
            scale = abs(u_gain) + abs(u_loss) + 1e-6
            EU = EU_raw / scale
            p_turn = expit(EU / beta)
            p_turn = np.clip(p_turn, eps, 1 - eps)

            max_k = trial_df['flip_number'].max()

            for _, row in trial_df.iterrows():
                is_last = (row['flip_number'] == max_k)
                if not is_last:
                    nll -= np.log(p_turn)
                else:
                    if popped:
                        nll -= np.log(p_turn)
                    else:
                        nll -= np.log(1 - p_turn)

        return nll

    def fit(self, flips_df, x0=None, bounds=None):
        x0 = [0.5, 1.0, 1.0]     # rho, lambda, beta
        bounds = [(0.01, 3.0), (0.01, 5.0), (0.01, 3.0)]
        res = minimize(self.nll, x0, args=(flips_df,),
                    bounds=bounds, method='L-BFGS-B')
        if not res.success:
            print("Fit failed:", res.message)
            return [np.nan, np.nan, np.nan]
        return res.x

    def simulate(self, params, flips_df, seed=None):
        rho, lam, beta = params
        rng = np.random.RandomState(seed)
        sim = flips_df.copy()
        sim['choice_sim'] = 0

        for trial, trial_df in sim.groupby('trial_number', sort=False):
            gain_amt = trial_df['gain_amount'].iloc[0]
            loss_amt = trial_df['loss_amount'].iloc[0]

            u_gain = self.utility(gain_amt, rho, lam)
            u_loss = self.utility(-loss_amt, rho, lam)

            EU_raw = 0.5 * u_gain + 0.5 * u_loss
            scale = abs(u_gain) + abs(u_loss) + 1e-6
            EU = EU_raw / scale
            p_turn = expit(EU / beta)

            for idx, row in trial_df.iterrows():
                sim.at[idx, 'choice_sim'] = rng.binomial(1, p_turn)
                if sim.at[idx, 'choice_sim'] == 0:
                    break  # останавливается

        return sim
    
    def predictive_check(self, user_flips):
        rho_fit, lam_fit, beta_fit = self.fit(user_flips)
        sim = self.simulate((rho_fit, lam_fit, beta_fit), user_flips, seed=42)

        real, simc = [], []
        for _, trial_df in user_flips.groupby('trial_number', sort=False):
            max_k = trial_df['flip_number'].max()
            for _, row in trial_df.iterrows():
                real.append(1 if (row['flip_number']<max_k or row['outcome']<0) else 0)
                simc.append(sim.loc[row.name,'choice_sim'])

        r2 = r2_score(real, simc)
        print("=== PPC CCT-HOT ===")
        print(f"rho={rho_fit:.3f}, λ={lam_fit:.3f}, β={beta_fit:.3f}, R² = {r2:.3f}")
        #return (rho_fit, lam_fit, beta_fit), pd.DataFrame({'choice_real': real, 'choice_sim': simc})
        return r2
    
    def parameter_recovery(self, template_flips, n_subjects=50):
        trues, fits = [], []
        for i in tqdm(range(n_subjects), desc="Recovery CCT-HOT"):
            rho_t = np.random.uniform(0.01,3)
            lam_t = np.random.uniform(0.01,10)
            beta_t= np.random.uniform(0.01,0.3)
            trues.append([rho_t, lam_t, beta_t])

            sim = self.simulate((rho_t, lam_t, beta_t), template_flips, seed=100+i)
            rho_f, lam_f, beta_f = self.fit(sim)
            fits.append([rho_f, lam_f, beta_f])

        cols = ['rho','lambda','beta']
        df = pd.DataFrame(
            np.hstack([trues, fits]),
            columns=[f"true_{c}" for c in cols] + [f"fit_{c}" for c in cols]
        )
        print("\n>>> Проверка на NaN и константы:")
        for c in cols:
            print(f"{c}:  std_true={np.std(df[f'true_{c}']):.4f},  std_fit={np.std(df[f'fit_{c}']):.4f},  NaNs={df[f'fit_{c}'].isna().sum()}")

        print("=== Parameter Recovery CCT-HOT ===")
        for c in cols:
            print(f"{c}: r = {np.corrcoef(df[f'true_{c}'], df[f'fit_{c}'])[0,1]:.3f}")
        return df
