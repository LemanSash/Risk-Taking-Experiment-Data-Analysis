import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from sklearn.metrics import r2_score
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

class COLDModel:
    """
    CCT-cold implementation of Prospect-Theory Model 1 (adapted from Wüllhorst et al.).
    - Cold design: participant chooses k (0..N) once per trial, no feedback.
    - N=32 for all trials (your design).
    - Utility and parameters correspond directly to Wüllhorst Model 1.
    """

    def __init__(self, data_df):
        self.data = data_df.copy().reset_index(drop=True)

    @staticmethod
    def log_comb(n, k):
        if (k < 0) or (k > n) or (n < 0):
            return -np.inf
        return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)

    @staticmethod
    def pt_utility(x, rho, lambd):
        """Prospect-theory utility (Eq.2 in Wüllhorst et al.)"""
        rho = float(np.clip(rho, 1e-6, 10.0))
        lambd = float(np.clip(lambd, 1e-6, 1e3))
        x = float(x)
        if x >= 0:
            return x ** rho
        else:
            return -lambd * (abs(x) ** rho)

    def utility(self, k, loss_cards, gain_amount, loss_amount, rho, lambd, N=32):
        """
        Expected utility for choosing k cards in cold CCT:
        EU(k) = p_no_loss * u(k*G) + (1 - p_no_loss) * u(loss_amount)
        """
        k = int(k)
        L = int(loss_cards)
        G = float(gain_amount)
        Loss = float(loss_amount)
        N = int(N)

        if k <= 0:
            return 0.0
        if k > N:
            return -1e9

        if k > (N - L):
            p_no_loss = 0.0
        else:
            log_p_no_loss = self.log_comb(N - L, k) - self.log_comb(N, k)
            if not np.isfinite(log_p_no_loss):
                p_no_loss = 0.0
            else:
                p_no_loss = float(np.exp(np.clip(log_p_no_loss, -700, 700)))

        p_no_loss = np.clip(p_no_loss, 0.0, 1.0)
        p_loss = 1.0 - p_no_loss

        u_gain = self.pt_utility(k * G, rho, lambd)
        u_loss = self.pt_utility(Loss, rho, lambd)

        eu = p_no_loss * u_gain + p_loss * u_loss
        return eu if np.isfinite(eu) else -1e9
    
    def utility_vec(self, ks, L, G, Loss, rho, lambd, N=32):
        """Векторная версия utility для массива ks"""
        ks = np.array(ks, dtype=int)
        N = int(N)
        L = int(L)
        G = float(G)
        Loss = float(Loss)

        u_gain = np.where(ks > 0, ks ** rho, 0.0)
        u_loss = -lambd * (Loss ** rho)

        # p_no_loss
        valid = ks <= (N - L)
        log_p_no_loss = np.full_like(ks, -np.inf, dtype=float)
        log_p_no_loss[valid] = (gammaln(N - L + 1) - gammaln(ks[valid] + 1) - gammaln(N - L - ks[valid] + 1)
                                - gammaln(N + 1) + gammaln(ks[valid] + 1) + gammaln(N - ks[valid] + 1))
        log_p_no_loss = np.clip(log_p_no_loss, -700, 700)
        p_no_loss = np.exp(log_p_no_loss)
        p_no_loss = np.clip(p_no_loss, 0.0, 1.0)
        p_loss = 1.0 - p_no_loss

        eu = p_no_loss * u_gain + p_loss * u_loss
        eu[ks <= 0] = 0.0
        eu[ks > N] = -1e9
        return eu

    def nll(self, params, data, N=32, use_lapse=False, lapse=0.0):
        """Векторная negative log-likelihood"""
        rho, lambd, beta = params
        if beta <= 0 or rho <= 0 or lambd <= 0:
            return 1e9

        k_range = np.arange(0, N + 1, dtype=int)
        total_log_lik = 0.0

        for _, trial in data.iterrows():
            N_trial = int(trial.get('total_cards', N))
            k_true = int(trial['num_cards'])
            L = int(trial['loss_cards'])
            G = float(trial['gain_amount'])
            Loss = float(trial['loss_amount'])

            ks = np.arange(0, N_trial + 1)
            eu_all = self.utility_vec(ks, L, G, Loss, rho, lambd, N_trial)
            logits = (eu_all - np.max(eu_all)) / beta
            logits = np.clip(logits, -700, 700)
            exp_logits = np.exp(logits)
            probs = exp_logits / np.sum(exp_logits)
            if use_lapse and lapse > 0:
                probs = (1 - lapse) * probs + lapse / probs.size

            prob_k = probs[k_true] if (0 <= k_true <= N_trial) else 1e-12
            total_log_lik += np.log(max(prob_k, 1e-12))

        return -total_log_lik

    def fit(self, user_data, N=32, n_starts=200, bounds=None, random_state=None,
            use_lapse=False, lapse=0.0, n_jobs=4):
        """Параллельный fit с ThreadPoolExecutor"""
        rng = np.random.default_rng(random_state)
        if bounds is None:
            bounds = [(0.01, 3.0), (0.01, 10.0), (0.01, 1.0)]

        inits = [
            (1.0, 1.5, 0.66),
            (0.5, 1.2, 0.3),
            (0.2, 2.0, 0.8),
        ]
        while len(inits) < n_starts:
            inits.append((
                float(rng.uniform(bounds[0][0], bounds[0][1])),
                float(rng.uniform(bounds[1][0], bounds[1][1])),
                float(rng.uniform(bounds[2][0], bounds[2][1]))
            ))

        def run_min(init):
            try:
                res = minimize(
                    lambda p: self.nll(p, user_data, N=N, use_lapse=use_lapse, lapse=lapse),
                    x0=np.array(init),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 1000}
                )
                return res if res.success else None
            except Exception:
                return None

        best_result = None
        best_fun = np.inf
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            for res in executor.map(run_min, inits):
                if res and res.fun < best_fun:
                    best_fun = res.fun
                    best_result = res

        if best_result is None:
            return {'rho': np.nan, 'lambda': np.nan, 'beta': np.nan, 'nll': np.nan}

        rho_hat, lambda_hat, beta_hat = best_result.x
        return {'rho': float(rho_hat), 'lambda': float(lambda_hat),
                'beta': float(beta_hat), 'nll': float(best_fun)}


    def simulate(self, params, user_data, N=32, seed=None, use_lapse=False, lapse=0.0):
        """Векторная симуляция выборов для всех trial сразу"""
        rng = np.random.default_rng(seed)
        rho, lambd, beta = params
        sims = []
        k_range_full = np.arange(0, N + 1, dtype=int)

        # Собираем параметры trial в массивы для векторной обработки
        Ls = user_data['loss_cards'].to_numpy(dtype=int)
        Gs = user_data['gain_amount'].to_numpy(dtype=float)
        Losses = user_data['loss_amount'].to_numpy(dtype=float)
        N_trials = user_data.get('total_cards', pd.Series(N, index=user_data.index)).to_numpy(dtype=int)

        for i in range(len(user_data)):
            k_range = np.arange(0, N_trials[i] + 1)
            eu_all = self.utility_vec(k_range, Ls[i], Gs[i], Losses[i], rho, lambd, N_trials[i])
            logits = (eu_all - np.max(eu_all)) / beta
            logits = np.clip(logits, -700, 700)
            exp_logits = np.exp(logits)
            probs = exp_logits / np.sum(exp_logits)
            if use_lapse and lapse > 0:
                probs = (1 - lapse) * probs + lapse / probs.size

            choice = int(rng.choice(k_range, p=probs))
            sims.append({
                'num_cards': choice,
                'gain_amount': Gs[i],
                'loss_amount': Losses[i],
                'loss_cards': Ls[i],
                'total_cards': N_trials[i]
            })

        return pd.DataFrame(sims)

    def predictive_check(self, user_data, N=32, nstarts=200):
        """Posterior predictive check (simulated vs real choices)."""
        fit_params = self.fit(user_data, N=N, n_starts=nstarts)
        params = (fit_params['rho'], fit_params['lambda'], fit_params['beta'])
        sim = self.simulate(params, user_data, N=N, seed=42)
        real = user_data['num_cards'].values
        simc = sim['num_cards'].values
        r2 = r2_score(real, simc)
        print("=== PPC CCT-COLD ===")
        print(f"rho={params[0]:.3f}, lambda={params[1]:.3f}, beta={params[2]:.3f}, R² = {r2:.3f}")
        return r2

    def parameter_recovery(self, template_data, n_subjects=50, N=32,
                           rho_range=(0.01, 3.0), lambda_range=(0.01, 10.0),
                           beta_range=(0.01, 1.0), n_starts=100, random_state=None):
        """
        Parameter recovery test: simulate N subjects with random true parameters,
        fit model to simulated data, compute correlations between true and recovered.
        """
        rng = np.random.default_rng(random_state)
        true_params = []
        fit_params = []

        for _ in tqdm(range(n_subjects), desc="Recovery CCT-COLD"):
            rho_true = rng.uniform(*rho_range)
            lambda_true = rng.uniform(*lambda_range)
            beta_true = rng.uniform(*beta_range)
            true_params.append([rho_true, lambda_true, beta_true])

            sim_data = self.simulate([rho_true, lambda_true, beta_true],
                                     template_data, N=N, seed=None)
            fitted = self.fit(sim_data, N=N, n_starts=n_starts)
            fit_params.append([fitted['rho'], fitted['lambda'], fitted['beta']])

        true_params = np.array(true_params)
        fit_params = np.array(fit_params)
        corrs = []
        for i, name in enumerate(['rho', 'lambda', 'beta']):
            if np.std(fit_params[:, i]) < 1e-8:
                r = np.nan
            else:
                r = np.corrcoef(true_params[:, i], fit_params[:, i])[0, 1]
            corrs.append((name, r))

        print(">>> Проверка на NaN и константы:")
        for i, name in enumerate(['rho', 'lambda', 'beta']):
            print(f"{name}: std_true={np.std(true_params[:, i]):.4f}, "
                  f"std_fit={np.std(fit_params[:, i]):.4f}, "
                  f"NaNs={np.isnan(fit_params[:, i]).sum()}")

        print("=== Parameter Recovery CCT-COLD ===")
        for name, r in corrs:
            print(f"{name}: r = {r:.3f}")

        return pd.DataFrame({
            'rho_true': true_params[:, 0], 'rho_fit': fit_params[:, 0],
            'lambda_true': true_params[:, 1], 'lambda_fit': fit_params[:, 1],
            'beta_true': true_params[:, 2], 'beta_fit': fit_params[:, 2]
        })
