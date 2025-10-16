import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import r2_score
import pymc as pm
import arviz as az
from numba import njit

@njit
def nll_numba(Arew, Apun, K0, betaF, betaP, decks, outcomes):
    """
    Быстрая версия отрицательного логарифма правдоподобия (NLL)
    без pandas, только NumPy.
    Полностью повторяет логику Haines et al. (2018).
    """
    K = 3.0 ** K0 - 1.0
    C = 3.0

    EV = np.zeros(4)
    EF = np.zeros(4)
    PS = np.zeros(4)
    nll = 0.0

    for t in range(len(decks)):
        deck = int(decks[t])
        x = float(outcomes[t])

        # integrated value V = EV + betaF*EF + betaP*PS
        V = EV + betaF * EF + betaP * PS
        V -= np.max(V)
        expV = np.exp(V)
        probs = expV / (np.sum(expV) + 1e-16)
        p_chosen = max(min(probs[deck], 1.0), 1e-12)
        nll -= np.log(p_chosen)

        # sign(x)
        s = 0.0
        if x > 0:
            s = 1.0
        elif x < 0:
            s = -1.0

        # EV update
        if x >= 0:
            EV[deck] += Arew * (x - EV[deck])
        else:
            EV[deck] += Apun * (x - EV[deck])

        # EF chosen
        if x >= 0:
            EF[deck] += Arew * (s - EF[deck])
        else:
            EF[deck] += Apun * (s - EF[deck])

        # EF unchosen
        if x >= 0:
            for j in range(4):
                if j == deck:
                    continue
                EF[j] += Apun * ((-s / C) - EF[j])
        else:
            for j in range(4):
                if j == deck:
                    continue
                EF[j] += Arew * ((-s / C) - EF[j])

        # PS update
        denom = 1.0 + K
        PS /= denom
        PS[deck] = 1.0 / denom

    return nll


class ORLModel:
    """
    Outcome-Representation Learning (ORL) model (Haines et al., 2018)
    с параметризацией K = 3**K0 - 1 и иерархическим байесовским фиттом (PyMC).
    """

    def __init__(self, data_df=None):
        self.data = None if data_df is None else data_df.copy()

    # ======== базовые утилиты ========
    @staticmethod
    def _softmax(values):
        """Softmax rule:
        p_j(t) = exp(V_j / τ) / sum_k exp(V_k / τ), with τ=1 (fixed).
        """
        v = np.asarray(values, dtype=float)
        v = v - np.max(v)
        ex = np.exp(v)
        probs = ex / (np.sum(ex) + 1e-16)
        return probs

    @staticmethod
    def _sgn(x):
        """sign function returning +1/-1/0 as in Haines et al. (2018)."""
        return 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)

    @staticmethod
    def _inv_logit(u):
        # numerically stable sigmoid
        return 1.0 / (1.0 + np.exp(-u))

    @staticmethod
    def _transform_from_unconstrained(u):
        """
        Преобразуем вектор неограниченных параметров u -> ограниченные Arew, Apun, K0, betaF, betaP
        u: array-like length 5
        трансформации:
          A = sigmoid(u0) in (0,1)
          Apun = sigmoid(u1) in (0,1)
          K0 = 5 * sigmoid(u2) in (0,5)
          betaF = tanh(u3) in (-1,1)
          betaP = tanh(u4) in (-1,1)
        """
        u = np.asarray(u, dtype=float)
        Arew = ORLModel._inv_logit(u[0])
        Apun = ORLModel._inv_logit(u[1])
        K0 = 5.0 * ORLModel._inv_logit(u[2])
        betaF = np.tanh(u[3])
        betaP = np.tanh(u[4])
        return np.array([Arew, Apun, K0, betaF, betaP], dtype=float)

    @staticmethod
    def nll_constrained(params_constrained, data, prior_beta_sigma=1.0):
        """
        Original NLL that expects constrained params:
        params_constrained = [Arew, Apun, K0, betaF, betaP]
        Adds small gaussian penalty on betas to stabilize optimisation (equivalent to weak prior).
        """
        Arew, Apun, K0, betaF, betaP = params_constrained
        # enforce numeric bounds for safety
        Arew = float(np.clip(Arew, 1e-8, 1.0-1e-8))
        Apun = float(np.clip(Apun, 1e-8, 1.0-1e-8))
        K0 = float(np.clip(K0, 0.0, 5.0))
        betaF = float(np.clip(betaF, -0.999999, 0.999999))
        betaP = float(np.clip(betaP, -0.999999, 0.999999))

        K = 3.0 ** K0 - 1.0

        EV = np.zeros(4, dtype=float)
        EF = np.zeros(4, dtype=float)
        PS = np.zeros(4, dtype=float)
        nll = 0.0
        C = 3.0

        for _, row in data.reset_index(drop=True).iterrows():
            deck = int(row['deck_num'])
            x = float(row['points_earned'])
            V = EV + betaF * EF + betaP * PS
            # softmax (temperature τ = 1)
            v = V - np.max(V)
            ex = np.exp(v)
            probs = ex / (np.sum(ex) + 1e-16)
            p_chosen = np.clip(probs[deck], 1e-12, 1.0)
            nll -= np.log(p_chosen)

            s = 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)

            # EV update
            if x >= 0:
                EV[deck] += Arew * (x - EV[deck])
            else:
                EV[deck] += Apun * (x - EV[deck])

            # EF chosen
            if x >= 0:
                EF[deck] += Arew * (s - EF[deck])
            else:
                EF[deck] += Apun * (s - EF[deck])

            # EF unchosen
            if x >= 0:
                for j in range(4):
                    if j == deck: continue
                    EF[j] += Apun * ((-s / C) - EF[j])
            else:
                for j in range(4):
                    if j == deck: continue
                    EF[j] += Arew * ((-s / C) - EF[j])

            # PS update
            denom = 1.0 + K
            PS = PS / denom
            PS[deck] = 1.0 / denom

        # regularization (weak gaussian prior on betas to avoid hitting +/-1 boundary)
        # penalty = (betaF^2 + betaP^2) / (2 * sigma^2)
        penalty = 0.5 * ((betaF ** 2 + betaP ** 2) / (prior_beta_sigma ** 2))
        nll += penalty
        return float(nll)

    @staticmethod
    def nll_unconstrained(u, data, prior_beta_sigma=1.0):
        """
        Wrapper for optimizer: accepts unconstrained u, transforms to constrained params and returns nll.
        """
        params = ORLModel._transform_from_unconstrained(u)
        return ORLModel.nll_constrained(params, data, prior_beta_sigma=prior_beta_sigma)


    def fit(self, user_data, n_restarts=5, x0_unconstrained=None, prior_beta_sigma=1.0):
        """
        Fit per-subject using unconstrained parametrization + multiple random restarts.
        Returns constrained params + best_nll.
        """
        data = user_data.copy().sort_values('trial_number').reset_index(drop=True)
        # default starting point in unconstrained space -> corresponds to A~0.3, Apun~0.2, K0~0.5, betaF~0.2, betaP~0.5
        if x0_unconstrained is None:
            # invert transforms to get unconstrained initial u0
            def inv_sig(p): return np.log(p / (1.0 - p))
            def inv_tanh(b): return np.arctanh(np.clip(b, -0.999999, 0.999999))
            A0 = 0.3; Ap0 = 0.2; K0_0 = 0.5; bF0 = 0.2; bP0 = 0.5
            u0 = np.array([inv_sig(A0), inv_sig(Ap0), inv_sig(K0_0/5.0), inv_tanh(bF0), inv_tanh(bP0)], dtype=float)
        else:
            u0 = np.asarray(x0_unconstrained, dtype=float)

        best_fun = np.inf
        best_u = None
        rng = np.random.default_rng(1234)

        # create many restarts in unconstrained space
        inits = [u0]
        for _ in range(max(0, n_restarts - 1)):
            # sample reasonable initial unconstrained values (normal around u0)
            init = u0 + rng.normal(scale=1.0, size=u0.shape)
            inits.append(init)

        for start in inits:
            try:
                # res = minimize(lambda uu: ORLModel.nll_unconstrained(uu, data, prior_beta_sigma=prior_beta_sigma),
                #                start, method='L-BFGS-B', options={'maxiter': 1000})
                decks = data['deck_num'].to_numpy(dtype=np.int64)
                outcomes = data['points_earned'].to_numpy(dtype=np.float64)

                res = minimize(
                    lambda uu: nll_numba(
                        *ORLModel._transform_from_unconstrained(uu), 
                        decks, outcomes
                    ),
                    start, method='L-BFGS-B', options={'maxiter': 1000}
                )

            except Exception:
                continue
            if res.success and res.fun < best_fun:
                best_fun = float(res.fun)
                best_u = res.x.copy()

        if best_u is None:
            # final fallback: try simple constrained fit on the transformed scale (older method)
            # produce NaNs if fails
            return np.array([np.nan]*5), np.nan

        best_params = ORLModel._transform_from_unconstrained(best_u)
        return best_params, best_fun

    # ======== Hierarchical Bayesian fit ========
    def fit_hierarchical(self, data_by_subject, draws=1000, chains=2):
        """
        Иерархический байесовский фит всех пользователей (PyMC).
        data_by_subject — dict: user_id -> DataFrame
        Возвращает ArviZ trace.
        """
        subjects = list(data_by_subject.keys())
        n_subj = len(subjects)

        with pm.Model() as model:
            # group-level (hyperpriors)
            mu_Arew = pm.Beta('mu_Arew', 2, 2)
            sigma_Arew = pm.Exponential('sigma_Arew', 2)

            mu_Apun = pm.Beta('mu_Apun', 2, 2)
            sigma_Apun = pm.Exponential('sigma_Apun', 2)

            mu_K0 = pm.TruncatedNormal('mu_K0', mu=1.5, sigma=1.0, lower=0, upper=5)
            sigma_K0 = pm.Exponential('sigma_K0', 2)

            mu_bF = pm.Normal('mu_bF', 0, 0.5)
            sigma_bF = pm.Exponential('sigma_bF', 2)

            mu_bP = pm.Normal('mu_bP', 0, 0.5)
            sigma_bP = pm.Exponential('sigma_bP', 2)

            # subject-level
            Arew = pm.Beta('Arew', alpha=mu_Arew * 10, beta=(1 - mu_Arew) * 10, shape=n_subj)
            Apun = pm.Beta('Apun', alpha=mu_Apun * 10, beta=(1 - mu_Apun) * 10, shape=n_subj)
            K0 = pm.TruncatedNormal('K0', mu=mu_K0, sigma=sigma_K0, lower=0, upper=5, shape=n_subj)
            bF = pm.Normal('bF', mu=mu_bF, sigma=sigma_bF, shape=n_subj)
            bP = pm.Normal('bP', mu=mu_bP, sigma=sigma_bP, shape=n_subj)

            # per-subject likelihood
            nlls = []
            for i, uid in enumerate(subjects):
                df = data_by_subject[uid]
                nll_i = ORLModel.nll([Arew[i], Apun[i], K0[i], bF[i], bP[i]], df)
                nlls.append(nll_i)
            total_nll = pm.Deterministic('total_nll', sum(nlls))
            pm.Potential('likelihood', -total_nll)

            trace = pm.sample(draws=draws, chains=chains, target_accept=0.9)

        return trace

    # ======== Simulation (random payoffs) ========
    def simulate(self, params, n_trials, seed=None):
        rng = np.random.default_rng(seed)
        Arew, Apun, K0, betaF, betaP = params
        Arew = np.clip(Arew, 0.0, 1.0)
        Apun = np.clip(Apun, 0.0, 1.0)
        K = 3 ** np.clip(K0, 0.0, 5.0) - 1
        betaF = np.clip(betaF, -1.0, 1.0)
        betaP = np.clip(betaP, -1.0, 1.0)

        EV = np.zeros(4)
        EF = np.zeros(4)
        PS = np.zeros(4)
        C = 3.0
        rows = []

        for t in range(1, n_trials + 1):
            V = EV + betaF * EF + betaP * PS
            probs = ORLModel._softmax(V)
            deck = int(rng.choice(4, p=probs))

            # random payoff generation
            if deck in [0, 1]:
                x = rng.choice(np.concatenate([np.repeat(100., 8), [-150., -250.]]))
            else:
                x = rng.choice(np.concatenate([np.repeat(50., 8), [-50., -100.]]))

            rows.append({'trial_number': t, 'deck_num': deck, 'points_earned': x})

            s = ORLModel._sgn(x)
            EV[deck] += (Arew if x >= 0 else Apun) * (x - EV[deck])

            if x >= 0:
                EF[deck] += Arew * (s - EF[deck])
            else:
                EF[deck] += Apun * (s - EF[deck])

            if x >= 0:
                for j in range(4):
                    if j != deck:
                        EF[j] += Apun * ((-s / C) - EF[j])
            else:
                for j in range(4):
                    if j != deck:
                        EF[j] += Arew * ((-s / C) - EF[j])

            denom = 1.0 + K
            PS /= denom
            PS[deck] = 1.0 / denom

        return pd.DataFrame(rows)

    # ======== Predictive check ========
    def predictive_check(self, real_data, seed=42):
        params, _ = self.fit(real_data)
        sim = self.simulate(params, n_trials=len(real_data), seed=seed)

        def adv_rates(df, n_blocks=5):
            df2 = df.sort_values('trial_number').reset_index(drop=True)
            n = len(df2)
            block_size = int(np.ceil(n / n_blocks))
            rates = []
            for b in range(n_blocks):
                block = df2.iloc[b*block_size: min((b+1)*block_size, n)]
                rate = np.mean(block['deck_num'].isin([2, 3]))
                rates.append(rate)
            return np.array(rates)

        real_vec = adv_rates(real_data)
        sim_vec = adv_rates(sim)
        r2 = r2_score(real_vec, sim_vec)

        diag = pd.DataFrame({
            'block': np.arange(1, len(real_vec)+1),
            'real_rate': real_vec,
            'sim_rate': sim_vec
        })
        print(diag)
        print(f"Params: Arew={params[0]:.3f}, Apun={params[1]:.3f}, K0={params[2]:.3f}, betaF={params[3]:.3f}, betaP={params[4]:.3f}")
        print(f"R² = {r2:.3f}")
        return r2, params, diag

    # ======== Parameter recovery ========
    def parameter_recovery(self, n_subjects=30, n_trials=200, seed0=123):
        rng = np.random.default_rng(seed0)
        true_params = []
        recovered = []

        for i in range(n_subjects):
            p_true = [
                rng.uniform(0.05, 0.95),  # Arew
                rng.uniform(0.05, 0.95),  # Apun
                rng.uniform(0.1, 4.5),    # K0
                rng.uniform(-0.9, 0.9),   # betaF
                rng.uniform(-0.9, 0.9)    # betaP
            ]
            sim = self.simulate(p_true, n_trials=n_trials, seed=seed0+i)
            p_fit, _ = self.fit(sim)
            true_params.append(p_true)
            recovered.append(p_fit)

        true_arr = np.array(true_params)
        rec_arr = np.array(recovered)
        cols = ['Arew','Apun','K0','betaF','betaP']
        df = pd.DataFrame(np.hstack([true_arr, rec_arr]),
                          columns=[f"true_{c}" for c in cols] + [f"fit_{c}" for c in cols])
        print("Parameter Recovery: correlations (true vs fit)")
        for c in cols:
            corr = np.corrcoef(df[f'true_{c}'], df[f'fit_{c}'])[0,1]
            print(f"{c}: r = {corr:.3f}")
        return df
