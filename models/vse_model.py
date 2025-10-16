import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import r2_score

class VSEModel:
    """
    Value plus Sequential Exploration (VSE) model
    Реализация по Ligneul, 2019.

    Параметры:
        [theta, delta, alpha, phi, consistency]

    theta ∈ [0,1]        – чувствительность к выигрышу/потере (Eq.1)
    delta ∈ [0,1]        – забывание exploitation (Eq.2)
    alpha ∈ [0,1]        – скорость возврата exploration к φ (Eq.3)
    phi ∈ (-∞,+∞)        – exploration bonus (Eq.3)
    consistency ∈ [0,5]  – детерминированность выбора (Eq.4)
    """

    def __init__(self, data_df=None):
        self.data = None if data_df is None else data_df.copy()

    # ---------- Softmax ----------
    @staticmethod
    def _safe_softmax(values, consistency):
        """Softmax с параметром consistency (не beta!)"""
        v = np.asarray(values, dtype=float) * float(consistency)
        v = v - np.max(v)  # стабилизация
        ex = np.exp(v)
        probs = ex / (np.sum(ex) + 1e-16)
        return probs

    # ---------- NLL ----------
    @staticmethod
    def nll(params, data):
        theta, delta, alpha, phi, consistency = params
        theta = np.clip(theta, 0.0, 1.0)
        delta = np.clip(delta, 0.0, 1.0)
        alpha = np.clip(alpha, 0.0, 1.0)
        consistency = np.clip(consistency, 0.0, 5.0)

        Exploit = np.zeros(4, dtype=float)
        Explore = np.ones(4, dtype=float)
        nll = 0.0

        for _, row in data.iterrows():
            deck = int(row['deck_num'])
            reward = float(row['points_earned'])

            # === Eq.4: вероятности выбора ===
            V = Exploit + Explore
            probs = VSEModel._safe_softmax(V, consistency)
            p_chosen = np.clip(probs[deck], 1e-12, 1.0)
            nll -= np.log(p_chosen)

            # === Eq.1: субъективная ценность ===
            gain = max(reward, 0.0)
            loss = max(-reward, 0.0)
            v = (gain ** theta) - (loss ** theta)

            # === Eq.2: обновление Exploit ===
            Exploit = delta * Exploit
            Exploit[deck] += v

            # === Eq.3: обновление Explore ===
            for d in range(4):
                if d == deck:
                    Explore[d] = 0.0
                else:
                    Explore[d] = Explore[d] + alpha * (phi - Explore[d])

        return float(nll)

    # ---------- FIT ----------
    def fit(self, user_data, n_restarts=10, x0=None, bounds=None):
        data = user_data.copy().sort_values(by=['trial_number']).reset_index(drop=True)
        if x0 is None:
            x0 = [0.5, 0.8, 0.3, 0.5, 1.0]
        if bounds is None:
            bounds = [
                (0.0, 1.0),   # theta
                (0.0, 1.0),   # delta
                (0.0, 1.0),   # alpha
                (-20.0, 20.0),# phi
                (0.0, 5.0)    # consistency
            ]

        best_res = None
        best_fun = np.inf
        rng = np.random.RandomState(0)
        inits = [np.array(x0, dtype=float)]
        for _ in range(max(0, n_restarts - 1)):
            init = np.array([rng.uniform(lo, hi) for (lo, hi) in bounds])
            inits.append(init)

        for start in inits:
            try:
                res = minimize(lambda p: self.nll(p, data), start,
                               method='L-BFGS-B', bounds=bounds)
            except Exception:
                continue
            if res.success and res.fun < best_fun:
                best_fun = float(res.fun)
                best_res = res

        if best_res is None:
            return np.array([np.nan] * 5), np.nan

        params = np.array(best_res.x, dtype=float)
        return params, float(best_fun)

    # ---------- SIMULATION ----------
    def simulate(self, params, n_trials, seed=None, payoff_dict=None):
        rng = np.random.RandomState(seed)
        theta, delta, alpha, phi, consistency = params
        consistency = np.clip(consistency, 0.0, 5.0)

        Exploit = np.zeros(4, dtype=float)
        Explore = np.ones(4, dtype=float)
        sim_rows = []

        for t in range(n_trials):
            V = Exploit + Explore
            probs = self._safe_softmax(V, consistency)
            deck = int(rng.choice(4, p=probs))

            # --- получение вознаграждения ---
            if payoff_dict and deck in payoff_dict:
                reward = float(rng.choice(payoff_dict[deck]))
            else:
                if deck in [0, 1]:
                    reward = float(rng.choice([100]*8 + [-150, -250]))
                else:
                    reward = float(rng.choice([50]*8 + [-50, -100]))

            sim_rows.append({
                'trial_number': t + 1,
                'deck_num': deck,
                'points_earned': reward
            })

            # --- Eq.1 ---
            gain = max(reward, 0.0)
            loss = max(-reward, 0.0)
            v = (gain ** theta) - (loss ** theta)

            # --- Eq.2 ---
            Exploit = delta * Exploit
            Exploit[deck] += v

            # --- Eq.3 ---
            for d in range(4):
                if d == deck:
                    Explore[d] = 0.0
                else:
                    Explore[d] = Explore[d] + alpha * (phi - Explore[d])

        return pd.DataFrame(sim_rows)

    # ---------- PPC ----------
    def predictive_check(self, real_data, seed=42):
        """
        Posterior predictive check (PPC) — сравнение последовательности реальных и сгенерированных паттернов SE.
        """
        params, nll = self.fit(real_data, n_restarts=12)
        sim = self.simulate(params, n_trials=len(real_data), seed=seed)

        def seq_vector(df):
            df = df.copy().reset_index(drop=True)
            seq_counts = np.zeros(4, dtype=float)
            denom = np.zeros(4, dtype=float)
            for i in range(1, len(df)):
                prev = int(df.loc[i-1, 'deck_num'])
                cur = int(df.loc[i, 'deck_num'])
                denom[prev] += 1
                if cur == (prev + 1) % 4:
                    seq_counts[prev] += 1
            with np.errstate(divide='ignore', invalid='ignore'):
                rates = np.where(denom > 0, seq_counts / denom, 0.0)
            return rates

        real_vec = seq_vector(real_data.sort_values('trial_number'))
        sim_vec = seq_vector(sim.sort_values('trial_number'))
        r2 = r2_score(real_vec, sim_vec)

        diag = pd.DataFrame({
            'prev_deck': [0, 1, 2, 3],
            'real_seq_rate': real_vec,
            'sim_seq_rate': sim_vec
        })

        print("\n=== VSE Posterior Predictive Check ===")
        print("Sequential exploration rates (prev -> next == prev+1):")
        print(diag)
        print(f"\nFitted params: θ={params[0]:.3f}, Δ={params[1]:.3f}, α={params[2]:.3f}, φ={params[3]:.3f}, C={params[4]:.3f}")
        print(f"R²(real vs sim seq-rates): {r2:.4f}")

        return r2, params, diag

    # ---------- PARAMETER RECOVERY ----------
    def parameter_recovery(self, n_subjects=50, n_trials=200, seed0=100):
        """
        Тест восстановления параметров (parameter recovery)
        """
        rng = np.random.RandomState(seed0)
        true_params = []
        recovered = []

        for i in range(n_subjects):
            p_true = [
                rng.rand(),        # theta
                rng.rand(),        # delta
                rng.rand(),        # alpha
                rng.uniform(-5, 5),# phi
                rng.uniform(0, 5)  # consistency
            ]
            sim = self.simulate(p_true, n_trials=n_trials, seed=seed0 + i)
            p_fit, _ = self.fit(sim, n_restarts=8)
            true_params.append(p_true)
            recovered.append(p_fit)

        true_arr = np.array(true_params)
        rec_arr = np.array(recovered)
        cols = ['theta','delta','alpha','phi','consistency']
        df = pd.DataFrame(
            np.hstack([true_arr, rec_arr]),
            columns=[f"true_{c}" for c in cols] + [f"fit_{c}" for c in cols]
        )

        print("\n=== Parameter Recovery (VSE) ===")
        for c in cols:
            if np.nanstd(df[f'fit_{c}']) == 0:
                corr = np.nan
            else:
                corr = np.corrcoef(df[f'true_{c}'], df[f'fit_{c}'])[0,1]
            print(f"{c:>12}: r = {np.round(corr,3) if not np.isnan(corr) else 'nan'}")
        return df
