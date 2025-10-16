import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import r2_score
import pandas as pd
from tqdm import tqdm
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed

class HOTModel:
    """
    Модель Prospect Theory для CCT-hot.
    """
    def __init__(self, data_df):
        self.data = data_df.copy()

    @staticmethod
    def utility(x, rho, lam):
        """
        Prospect‐theory utility.
        """
        x = np.asarray(x)
        out = np.empty_like(x, dtype=float)
        pos_mask = x >= 0
        neg_mask = ~pos_mask
        out[pos_mask] = np.power(x[pos_mask], rho)
        out[neg_mask] = -lam * np.power(np.abs(x[neg_mask]), rho)
        return out
    

    def nll(self, params, flips_df):
        """
        Negative log-likelihood строго по Model 1 (векторизовано).
        u(x) = x^rho (x>=0), u(x) = -lambda*(-x)^rho (x<0)
        EU = (1 - p_loss)*u(gain) + p_loss*u(loss)
        P(continue) = sigmoid(EU / beta)
        """
        rho, lam, beta = params
        eps = 1e-12

        if 'choice' not in flips_df.columns:
            raise ValueError("nll ожидает колонку 'choice' (1=continue, 0=stop) в flips_df.")

        nll = 0.0

        # Группировка по trial_number — как и раньше
        for _, trial_df in flips_df.groupby('trial_number', sort=False):
            gain_amt = trial_df['gain_amount'].iloc[0]
            loss_amt = trial_df['loss_amount'].iloc[0]

            # Если loss_amount хранится как положительное число (например 70),
            # то u_loss = self.utility(-loss_amt, rho, lam)
            # Если уже отрицательное (например -70) — тогда просто self.utility(loss_amt, rho, lam)
            if loss_amt < 0:
                u_loss = self.utility(loss_amt, rho, lam)
            else:
                u_loss = self.utility(-loss_amt, rho, lam)
            u_gain = self.utility(gain_amt, rho, lam)

            # Преобразуем в NumPy и считаем всё векторно
            flip_no = trial_df['flip_number'].to_numpy()
            loss_cards = trial_df['loss_cards'].to_numpy()
            choice = trial_df['choice'].to_numpy()

            denom = 32 - (flip_no - 1)   # при необходимости заменить 32 → row['total_cards']
            p_loss = loss_cards / denom
            EU = (1 - p_loss) * u_gain + p_loss * u_loss

            # Вероятность continue по модели
            p_turn = expit(EU / beta)
            p_turn = np.clip(p_turn, eps, 1 - eps)

            # Вычисляем отрицательный логарифм правдоподобия для всех строк за раз
            nll_trial = -np.sum(choice * np.log(p_turn) + (1 - choice) * np.log(1 - p_turn))
            nll += nll_trial

        return nll


    def fit(self, flips_df, bounds=None, n_starts=200):
        if bounds is None:
            bounds = [(0.01,3.0), (0.01,10.0), (0.01,1.0)]  # rho, lambda, beta per paper
        best_res = None
        for i in range(n_starts):
            x0 = [np.random.uniform(b[0], b[1]) for b in bounds]
            res = minimize(self.nll, x0, args=(flips_df,), bounds=bounds, method='L-BFGS-B')
            if res.success and (best_res is None or res.fun < best_res.fun):
                best_res = res
        if best_res is None:
            return [np.nan, np.nan, np.nan]
        return best_res.x

    
    def fit_with_choice(self, flips_df, x0=None, bounds=None):
        """
        То же, что fit; оставлена для совместимости с parameter_recovery.
        """
        return self.fit(flips_df, x0=x0, bounds=bounds)


    def simulate(self, params, flips_df, seed=None):
        rho, lam, beta = params
        rng = np.random.RandomState(seed)
        
        sim = flips_df.drop(columns=['choice'], errors='ignore').copy()
        sim['choice'] = 0 

        for _, trial_df in sim.groupby('trial_number', sort=False):
            gain_amt = trial_df['gain_amount'].iloc[0]
            loss_amt = trial_df['loss_amount'].iloc[0]

            u_gain = self.utility(gain_amt, rho, lam)
            u_loss = self.utility(-loss_amt, rho, lam)

            for j in trial_df.index:
                row = sim.loc[j]
                denom = 32 - (row['flip_number'] - 1)
                p_loss = row['loss_cards'] / denom

                EU = (1 - p_loss) * u_gain + p_loss * u_loss
                #p_turn = expit(beta * EU)
                p_turn = expit(EU / beta)
                p_turn = np.clip(p_turn, 1e-12, 1 - 1e-12)

                sim.at[j, 'choice'] = int(rng.binomial(1, p_turn))

        sim['choice'] = sim['choice'].astype(int)
        return sim


    def predictive_check(self, user_flips):
        rho_fit, lam_fit, beta_fit = self.fit(user_flips)

        sim = self.simulate((rho_fit, lam_fit, beta_fit), user_flips, seed=42)

        real = (
            user_flips['choice']
            .astype(int)
            .reindex(sim.index)  
            .fillna(0)
            .to_numpy()
        )
        simc = sim['choice'].astype(int).to_numpy()

        r2 = r2_score(real, simc)
        print("=== PPC CCT-HOT ===")
        print(f"rho={rho_fit:.3f}, λ={lam_fit:.3f}, β={beta_fit:.3f}, R² = {r2:.3f}")
        return r2


    def parameter_recovery(self, template_flips, n_subjects=50, n_starts=20, n_jobs=8):
        """
        Параллельный Parameter Recovery (через потоки).
        """
        rng = np.random.default_rng(123)
        true_params = np.column_stack([
            rng.uniform(0.01, 3.0, n_subjects),   # rho
            rng.uniform(0.01, 10.0, n_subjects),  # lambda
            rng.uniform(0.01, 1.0, n_subjects)    # beta
        ])

        def one_recovery(i, params):
            rho_t, lam_t, beta_t = params
            sim = self.simulate((rho_t, lam_t, beta_t), template_flips, seed=100+i)
            rho_f, lam_f, beta_f = self.fit(sim, n_starts=n_starts)
            return [rho_t, lam_t, beta_t, rho_f, lam_f, beta_f]

        results = []
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = {executor.submit(one_recovery, i, true_params[i]): i for i in range(n_subjects)}
            for f in tqdm(as_completed(futures), total=n_subjects, desc="Parallel recovery"):
                results.append(f.result())

        df = pd.DataFrame(results, columns=[
            "true_rho", "true_lambda", "true_beta",
            "fit_rho", "fit_lambda", "fit_beta"
        ])

        # отчёт
        print("\n=== Parameter Recovery CCT-HOT ===")
        for c in ["rho", "lambda", "beta"]:
            if df[f'fit_{c}'].isna().all():
                print(f"{c}: r = nan (все NaN)")
            else:
                r = np.corrcoef(df[f'true_{c}'], df[f'fit_{c}'])[0, 1]
                print(f"{c}: r = {r:.3f}")

        return df
