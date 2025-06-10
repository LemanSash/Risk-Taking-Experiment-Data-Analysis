import numpy as np
from scipy.optimize import minimize
import pandas as pd
from sklearn.metrics import r2_score

class VSEModel:
    """
    Модель Value plus Sequential Exploration для IGT.
    """
    def __init__(self, data_df):
        self.data = data_df.copy()

    @staticmethod
    def nll(params, data):
        """
        Отрицательное логарифмическое правдоподобие VSE-модели.
        
        params = [alpha_pos, alpha_neg, beta, gamma]
        alpha_pos, alpha_neg — скорости обучения для положительных/отрицательных ошибок
        beta  — inverse temperature для софтмакс
        gamma — бонус последовательного исследования
        data — DataFrame с колонками ['deck_num', 'points_earned'], отсортированный по trial_number
        """
        alpha_pos, alpha_neg, beta, gamma = params
        
        Q = np.zeros(4)      # Exploit-значения (Value)
        nll = 0.0
        last_choice = None
        
        for _, row in data.iterrows():
            deck   = int(row['deck_num'])      # 0,1,2,3
            reward = row['points_earned']
            
            # 1) Sequential exploration bonus
            S = np.zeros(4)
            if last_choice is not None:
                next_deck = (last_choice + 1) % 4
                S[next_deck] = gamma
            
            # 2) Softmax over V = Q + S
            V = Q + S
            expV = np.exp(beta * V - np.max(beta * V))  # стабилизация
            probs = expV / np.sum(expV)
            
            # 3) accu nll
            nll -= np.log(probs[deck] + 1e-12)
            
            # 4) Update Q-value выбранной колоды
            pe = reward - Q[deck]
            alpha = alpha_pos if pe >= 0 else alpha_neg
            Q[deck] += alpha * pe
            
            last_choice = deck
        return nll

    def fit(self, user_data,  x0=None, bounds=None):
        """
        Подгонка параметров VSE-модели для одного пользователя.
        user_data: DataFrame с ['deck_num','points_earned','trial_number']
        """
        # исходные приближения
        x0 = [0.2, 0.2, 1.0, 0.5]  
        # границы: alpha∈[0,1], beta∈[0,10], gamma∈[0,10]
        bnds = [(0,1), (0,1), (0,10), (0,10)]
        
        res = minimize(self.nll, x0, args=(user_data,), 
                    method='L-BFGS-B', bounds=bnds)
        return res.x, res.fun

    def simulate(self, params, n_trials, seed=None):
        """Симуляция n_trials по VSE-модели с нулевым вознаграждением ~ Uniform(-100,100)"""
        if seed is not None:
            np.random.seed(seed)
        alpha_pos, alpha_neg, beta, gamma = params
        Q = np.zeros(4)
        last_choice = None

        sim = []
        for t in range(n_trials):
            # 1) выставляем бонус, считаем probs
            S = np.zeros(4)
            if last_choice is not None:
                S[(last_choice + 1) % 4] = gamma
            V = Q + S
            expV = np.exp(beta * V - np.max(beta * V))
            probs = expV / np.sum(expV)

            # 2) делаем выбор и генерируем «вознаграждение»
            deck = np.random.choice(4, p=probs)
            reward = np.random.choice([-100, -50, +50, +100])  # примерный IGT‐разброс

            sim.append({'trial': t, 'deck_num': deck, 'points_earned': reward})
            
            # 3) update
            pe = reward - Q[deck]
            alpha = alpha_pos if pe >= 0 else alpha_neg
            Q[deck] += alpha * pe
            last_choice = deck
        sim_df = pd.DataFrame(sim)
        return sim_df
    
    def predictive_check(self, real_data):
        fitted_params, _ = self.fit(real_data)  # ← учитываем, что возвращается (params, nll)
        sim_data = self.simulate(fitted_params, len(real_data), seed=42)

        real_counts = real_data['deck_num'].value_counts(normalize=True).sort_index()
        sim_counts  = sim_data ['deck_num'].value_counts(normalize=True).sort_index()
        
        df = pd.DataFrame({
            'real_freq': real_counts,
            'sim_freq' : sim_counts
        })
        r2 = r2_score(df.real_freq, df.sim_freq)
        print("PPC: сравнение частот выбора колод (реал vs. симуляция)")
        print(df)
        print(f"R² между real_freq и sim_freq: {r2:.3f}")
        return r2
    
    def parameter_recovery(self, n_subjects=50, n_trials=200):
        true_params = []
        recovered  = []

        for i in range(n_subjects):
            p_true = [
                np.random.rand(),       # alpha_pos
                np.random.rand(),       # alpha_neg
                np.random.rand() * 5,   # beta
                np.random.rand() * 5    # gamma
            ]
            
            sim = self.simulate(p_true, n_trials, seed=100+i)
            p_fit, _ = self.fit(sim)  # ← теперь получаем только fit-параметры
            true_params.append(p_true)
            recovered.append(p_fit)

        # Анализ результатов
        true_arr = np.array(true_params)
        rec_arr  = np.array(recovered)
        cols = ['alpha_pos','alpha_neg','beta','gamma']
        df = pd.DataFrame(
            np.hstack([true_arr, rec_arr]),
            columns=[f"true_{c}" for c in cols] + [f"fit_{c}" for c in cols]
        )
        print("Parameter Recovery: корреляции (true vs fit)")
        for c in cols:
            corr = np.corrcoef(df[f"true_{c}"], df[f"fit_{c}"])[0,1]
            print(f"  {c}: r = {corr:.3f}")
        return df