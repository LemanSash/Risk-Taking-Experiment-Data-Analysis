import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from sklearn.metrics import r2_score
import pandas as pd
from tqdm import tqdm

class COLDModel:
    """
    Модель Value plus Sequential Exploration для IGT.
    """
    def __init__(self, data_df):
        self.data = data_df.copy()

    @staticmethod
    def log_comb(n, k):
        """Логарифм биномиального коэффициента для больших чисел"""
        return gammaln(n+1) - gammaln(k+1) - gammaln(n-k+1)
    
    def utility(self, k, loss_cards, gain_amount, loss_amount, rho, lambd, N=32):
        loss_abs = abs(loss_amount)
        j_min = max(0, k - (N - loss_cards))
        j_max = min(k, loss_cards)
        
        if j_min > j_max:
            return 0.0
            
        total_eu = 0.0
        total_log_prob = -np.inf
        
        # Вычисляем логарифмы вероятностей
        log_probs = []
        for j in range(j_min, j_max + 1):
            try:
                log_p = (
                    self.log_comb(loss_cards, j) +
                    self.log_comb(N - loss_cards, k - j) -
                    self.log_comb(N, k)
                )
                log_probs.append(log_p)
            except:
                log_probs.append(-np.inf)
        
        # Нормализуем вероятности
        max_log = max(log_probs)
        if np.isinf(max_log):
            return 0.0
            
        norm_log_probs = log_probs - max_log
        probs = np.exp(norm_log_probs)
        total_prob = np.sum(probs)
        if total_prob <= 0:
            return 0.0
            
        probs /= total_prob

        for idx, j in enumerate(range(j_min, j_max + 1)):
            win_cards = k - j
            outcome = win_cards * gain_amount - j * loss_abs
            
            # Вычисление полезности
            if outcome >= 0:
                utility = outcome ** rho
            else:
                utility = -lambd * (abs(outcome) ** rho)
            
            total_eu += probs[idx] * utility

        return total_eu
    
    def nll(self, params, data, N=32):
        rho, lambd, beta = params
        total_log_lik = 0.0
        
        for _, trial in data.iterrows():
            k_true = int(trial['num_cards'])
            L = int(trial['loss_cards'])
            G = trial['gain_amount']
            loss_abs = abs(trial['loss_amount'])
            
            # Рассчитать EU для всех возможных k
            eu_all = []
            for k in range(0, N + 1):
                eu = self.utility(k, L, G, loss_abs, rho, lambd, N)
                eu_all.append(eu)
            eu_all = np.array(eu_all)
            
            # Стабилизированный softmax
            eu_all_shifted = eu_all - np.max(eu_all)
            exp_eu = np.exp(eu_all_shifted / beta)
            softmax_probs = exp_eu / np.sum(exp_eu)
            
            # Вероятность выбора k_true
            if k_true < 0 or k_true > N:
                prob_k_true = 1e-10
            else:
                prob_k_true = max(softmax_probs[k_true], 1e-10)
                
            total_log_lik += np.log(prob_k_true)
        
        return -total_log_lik  # Возвращаем negative log-likelihood

    def fit(self, user_data, N=32):
        initial_params = [0.5, 1.5, 0.5]  # [rho, lambda, beta]
        bounds = [(0.01, 3), (0.01, 10), (0.01, 10)]
        
        result = None
        try:
            result = minimize(
                self.nll,
                initial_params,
                args=(user_data, N),
                bounds=bounds,
                method='L-BFGS-B',
                options={'maxiter': 500}
            )
        except Exception as e:
            print(f"Ошибка минимизации: {str(e)}")
        
        if result and result.success:
            return {
                'cold_rho': result.x[0],
                'cold_lambda': result.x[1],
                'cold_beta': result.x[2],
            }
        else:
            error_msg = result.message if result else "Ошибка оптимизации"
            user_id = user_data['user_id'].iloc[0] if 'user_id' in user_data.columns else "unknown"
            print(f"Сбой для пользователя {user_id}: {error_msg}")
            return {
                'cold_rho': np.nan,
                'cold_lambda': np.nan,
                'cold_beta': np.nan,
            }
    
    def simulate(self, params, data, N=32, seed=None):
        rho, lambd, beta = params
        rng = np.random.RandomState(seed)
        sim_data = data.copy()
        sim_choices = []

        for _, trial in sim_data.iterrows():
            L = int(trial['loss_cards'])
            G = trial['gain_amount']
            loss_abs = abs(trial['loss_amount'])

            eu_all = []
            for k in range(N+1):
                eu = self.utility(k, L, G, loss_abs, rho, lambd, N)
                eu_all.append(eu)
            eu_all = np.array(eu_all)

            # Softmax для получения вероятностей
            eu_all_shifted = eu_all - np.max(eu_all)
            exp_eu = np.exp(eu_all_shifted / beta)
            softmax_probs = exp_eu / np.sum(exp_eu)

            # Генерация случайного выбора k
            choice = rng.choice(np.arange(N+1), p=softmax_probs)
            sim_choices.append(choice)

        sim_data['choice_sim'] = sim_choices
        return sim_data
    
    def predictive_check(self, user_data, N=32):
        # Фитируем модель по реальным данным
        fit_params = self.fit(user_data, N)
        params = (fit_params['cold_rho'], fit_params['cold_lambda'], fit_params['cold_beta'])

        sim = self.simulate(params, user_data, N=N, seed=42)

        # real и sim — сравнение по выбору k
        real = user_data['num_cards'].values
        simc = sim['choice_sim'].values

        # Можно посчитать R^2 или скор между числовыми выборками
        r2 = r2_score(real, simc)
        print("=== PPC CCT-COLD ===")
        print(f"rho={params[0]:.3f}, lambda={params[1]:.3f}, beta={params[2]:.3f}, R² = {r2:.3f}")

        #return params, pd.DataFrame({'choice_real': real, 'choice_sim': simc})
        return r2
    
    def parameter_recovery(self, template_data, n_subjects=50, N=32):
        trues, fits = [], []
        for i in tqdm(range(n_subjects), desc="Recovery CCT-COLD"):
            rho_t = np.random.uniform(0.01, 3)
            lam_t = np.random.uniform(0.01, 10)
            beta_t = np.random.uniform(0.01, 10)
            trues.append([rho_t, lam_t, beta_t])

            sim = self.simulate((rho_t, lam_t, beta_t), template_data, N=N, seed=100+i)
            fit_res = self.fit(sim, N)
            fits.append([fit_res['cold_rho'], fit_res['cold_lambda'], fit_res['cold_beta']])

        cols = ['rho', 'lambda', 'beta']
        df = pd.DataFrame(
            np.hstack([trues, fits]),
            columns=[f"true_{c}" for c in cols] + [f"fit_{c}" for c in cols]
        )
        
        print("\n>>> Проверка на NaN и константы:")
        for c in cols:
            print(f"{c}: std_true={np.std(df[f'true_{c}']):.4f}, std_fit={np.std(df[f'fit_{c}']):.4f}, NaNs={df[f'fit_{c}'].isna().sum()}")

        print("=== Parameter Recovery CCT-COLD ===")
        for c in cols:
            corr = np.corrcoef(df[f'true_{c}'], df[f'fit_{c}'])[0,1]
            print(f"{c}: r = {corr:.3f}")
        return df
    
