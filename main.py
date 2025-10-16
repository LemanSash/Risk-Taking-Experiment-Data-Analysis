import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor, as_completed
from models.ev_model import EVModel
from models.ew_model import EWModel
from models.ewmv_model import EWMVModel
from models.haffke_model_cold import HaffkeColdModel_Model3
from models.haffke_model_hot import HaffkeModel
from models.orl_model import ORLModel
from models.par4_model import Par4Model
from models.pvl_model import PVLModel
from models.stl_model import STLModel
from models.vpp_model import VPPModel
from models.wallsten_model import Model3_Wallsten
from models.wullhorst_model2_cold import COLDModel2
from models.wullhorst_model2_hot import HOTModel2
from models.vse_model import VSEModel
from models.stld_model import STLDModel
from models.wullhorst_model1_hot import HOTModel
from models.wullhorst_model1_cold import COLDModel


if __name__ == '__main__':
    # === Here you can find examples of using models ===
    igt_df = pd.DataFrame() # IGT df, the structure of the df for each model can be found in the model script file
    bart_df = pd.DataFrame() # BART df, the structure of the df for each model can be found in the model script file
    hot_df = pd.DataFrame() # CCT-hot df, the structure of the df for each model can be found in the model script file
    cold_df = pd.DataFrame() # CCT-cold df, the structure of the df for each model can be found in the model script file

    # === VSE ===
    vse = VSEModel(igt_df)
    vse_results = []
    for uid, df in igt_df.groupby('user_id'):
        df = df.sort_values('trial_number')
        if len(df) < 20:
            continue
        try:
            params, nll = vse.fit(df)
            vse_results.append({
                'user_id': uid,
                'igt_vse_alpha_pos': params[0],
                'igt_vse_alpha_neg': params[1],
                'igt_vse_beta': params[2],
                'igt_vse_gamma': params[3],
            })
        except Exception as e:
            print(f"Error with user_id={uid}: {e}")

    igt_params_df = pd.DataFrame(vse_results)
    print('IGT params df')
    print(igt_params_df.head())
    igt_params_df.to_csv('igt_vse.csv')

    # === EV ===
    ev_results = []
    for uid, df_user in igt_df.groupby('user_id'):
        df = df_user.sort_values('trial_number')
        if len(df) < 20:
            continue
        try:
            ev = EVModel(df)
            params, nll = ev.fit(df)
            ev_results.append({'user_id': uid, 'igt_ev_w': params[0], 'igt_ev_a': params[1], 'igt_ev_c': params[2], 'nll': nll})
        except Exception as e:
            print(f"Error with user {uid}: {e}")

    ev_params_df = pd.DataFrame(ev_results)
    print(ev_params_df.head())
    ev_params_df.to_csv('igt_ev.csv')

    # === PVL-Delta ===
    pvl_delta_results = []
    for uid, df_user in igt_df.groupby('user_id'):
        df = df_user.sort_values('trial_number')
        if len(df) < 20:
            continue
        try:
            pvl_delta = PVLModel(df, update_rule="delta")
            params, nll = pvl_delta.fit(df)
            pvl_delta_results.append({
                'user_id': uid,
                'igt_pvl_delta_A': params[0], 'igt_pvl_delta_alpha': params[1], 'igt_pvl_delta_lam': params[2], 'igt_pvl_delta_c': params[3], 'nll': nll
            })
        except Exception as e:
            print(f"Error with user {uid}: {e}")

    pvl_delta_params_df = pd.DataFrame(pvl_delta_results)
    print(pvl_delta_params_df.head())
    pvl_delta_params_df.to_csv('igt_pvl_delta.csv')

    # === PVL-Decay ===
    pvl_decay_results = []
    for uid, df_user in igt_df.groupby('user_id'):
        df = df_user.sort_values('trial_number')
        if len(df) < 20:
            continue
        try:
            pvl_decay = PVLModel(df, update_rule="decay")
            params, nll = pvl_decay.fit(df)
            pvl_decay_results.append({
                'user_id': uid,
                'igt_pvl_decay_A': params[0], 'igt_pvl_decay_alpha': params[1], 'igt_pvl_decay_lam': params[2], 'igt_pvl_decay_c': params[3], 'nll': nll
            })
        except Exception as e:
            print(f"Error with user {uid}: {e}")

    pvl_decay_params_df = pd.DataFrame(pvl_decay_results)
    print(pvl_decay_params_df.head())
    pvl_decay_params_df.to_csv('igt_pvl_delta.csv')

    # === VPP ===
    def fit_one(uid, df_user):
        df = df_user.sort_values('trial_number')
        vpp = VPPModel(df)
        params, nll = vpp.fit(df, n_restarts=5)
        return {
            'user_id': uid,
            'igt_vpp_phi': params[0], 
            'igt_vpp_alpha': params[1], 
            'igt_vpp_lam': params[2], 
            'igt_vpp_c': params[3],
            'igt_vpp_w': params[4], 
            'igt_vpp_K': params[5],
            'igt_vpp_eps_pos': params[6], 
            'igt_vpp_eps_neg': params[7], 
            'nll': nll
        }

    vpp_results = Parallel(n_jobs=8)(
        delayed(fit_one)(uid, df_user)
        for uid, df_user in igt_df.groupby('user_id')
    )
    vpp_params_df = pd.DataFrame(vpp_results)
    print(vpp_params_df.head())
    vpp_params_df.to_csv('igt_vpp.csv')

    # === ORL ===
    def fit_user(uid, df):
        orl = ORLModel(df)
        params, nll = orl.fit(df, n_restarts=5)
        return {
            'user_id': uid,
            'igt_orl_Arew': params[0], 'igt_orl_Apun': params[1],
            'igt_orl_K0': params[2], 'igt_orl_K': 3**params[2]-1,
            'igt_orl_betaF': params[3], 'igt_orl_betaP': params[4],
            'nll': nll
        }

    orl_results = Parallel(n_jobs=8)(delayed(fit_user)(uid, df) for uid, df in igt_df.groupby('user_id'))

    orl_params_df = pd.DataFrame(orl_results)
    print(orl_params_df.head())
    orl_params_df.to_csv('igt_orl.csv')

    # === STL-D ===
    stld = STLDModel(bart_df)
    stld_params_list = []

    for user_id in tqdm(bart_df['user_id'].unique(), desc="Fitting STL-D models"):
        user_df = bart_df[bart_df['user_id'] == user_id]
        if len(user_df) < 5:
            continue
        try:
            params = stld.fit(user_df)
            stld_params_list.append({
                'user_id': user_id,
                'bart_stl_decay_w1': params[0],
                'bart_stl_decay_vwin': params[1],
                'bart_stl_decay_vloss': params[2],
                'bart_stl_decay_alpha': params[3],
                'bart_stl_decay_beta': params[4]
            })
        except Exception as e:
            print(f"Ошибка для пользователя {user_id}: {e}")
    bart_params_df = pd.DataFrame(stld_params_list)
    print('BART params df')
    print(bart_params_df.head())
    bart_params_df.to_csv('bart_stl_decay.csv')

    # === STL ===
    model_stl = STLModel(bart_df)
    params_list_stl = []

    for user_id in tqdm(bart_df['user_id'].unique(), desc="Fitting STL models"):
        user_df = bart_df[bart_df['user_id'] == user_id]
        if len(user_df) < 5:
            continue
        try:
            params = model_stl.fit(user_df)
            params_list_stl.append({
                'user_id': user_id,
                'bart_stl_w1': params[0],
                'bart_stl_vwin': params[1],
                'bart_stl_vloss': params[2],
                'bart_stl_beta': params[3]
            })
        except Exception as e:
            print(f"Error with user {user_id}: {e}")
    params_df_stl = pd.DataFrame(params_list_stl)
    print('Параметры STL модели')
    print(params_df_stl.head())
    params_df_stl.to_csv('bart_stl.csv')

    # === Par4 ===
    model_par4 = Par4Model(bart_df)
    params_list_par4 = []

    for user_id in tqdm(bart_df['user_id'].unique(), desc="Fitting Par4 models"):
        user_df = bart_df[bart_df['user_id'] == user_id]
        if len(user_df) < 5:
            continue
        try:
            params = model_par4.fit(user_df)
            params_list_par4.append({
                'user_id': user_id,
                'bart_par4_risk': params[0],
                'bart_par4_consistency': params[1],
                'bart_par4_sensitivity': params[2],
                'bart_par4_bias': params[3]
            })
        except Exception as e:
            print(f"Error with user {user_id}: {e}")
    params_df_par4 = pd.DataFrame(params_list_par4)
    print('Параметры Par4 модели')
    print(params_df_par4.head())
    params_df_par4.to_csv('bart_par4.csv')

    # === EW ===
    model_ew = EWModel(bart_df)
    params_list_ew = []

    for user_id in tqdm(bart_df['user_id'].unique(), desc="Fitting EW models"):
        user_df = bart_df[bart_df['user_id'] == user_id]
        if len(user_df) < 5:
            continue
        try:
            params = model_ew.fit(user_df)
            params_list_ew.append({
                'user_id': user_id,
                'bart_ew_risk': params[0],
                'bart_ew_consistency': params[1]
            })
        except Exception as e:
            print(f"Error with user {user_id}: {e}")
    params_df_ew = pd.DataFrame(params_list_ew)
    print('Параметры Exponential Weighting модели')
    print(params_df_ew.head())
    params_df_ew.to_csv('bart_ew.csv')

    # === EWMV ===
    model_ewmv = EWMVModel(bart_df)
    params_list_ewmv = []

    for user_id in tqdm(bart_df['user_id'].unique(), desc="Fitting EWMV models"):
        user_df = bart_df[bart_df['user_id'] == user_id]
        if len(user_df) < 5:
            continue
        try:
            params = model_ewmv.fit(user_df)
            params_list_ewmv.append({
                'user_id': user_id,
                'bart_ewmv_risk': params[0],
                'bart_ewmv_consistency': params[1],
                'bart_ewmv_sensitivity': params[2],
                'bart_ewmv_bias': params[3]
            })
        except Exception as e:
            print(f"Error with user {user_id}: {e}")
    params_df_ewmv = pd.DataFrame(params_list_ewmv)
    print('Параметры Exponential Weighting Mean-Variance модели')
    print(params_df_ewmv.head())
    params_df_ewmv.to_csv('bart_ewmv.csv')

    # === Wallsten Model 3 ===
    model_wallsten = Model3_Wallsten(reward_per_pump=5, max_pumps=128)
    params_list_model3 = []

    for user_id in tqdm(bart_df['user_id'].unique(), desc="Fitting Model 3 (Wallsten)"):
        user_df = bart_df[bart_df['user_id'] == user_id]
        if len(user_df) < 5:
            continue
        try:
            res = model_wallsten.fit(user_df)
            if res is not None:
                params_list_model3.append({
                    'user_id': user_id,
                    'bart_wallsten_gamma_plus': res.x[0],
                    'bart_wallsten_beta': res.x[1],
                    'bart_wallsten_a0': res.x[2],
                    'bart_wallsten_m0': res.x[3],
                    'nll': res.fun
                })
            else:
                print(f"Fit failed for user {user_id}")
        except Exception as e:
            print(f"Ошибка для user_id={user_id}, шар=blue: {e}")

    params_df_model3 = pd.DataFrame(params_list_model3)
    print("Параметры модели 3 (Wallsten):")
    print(params_df_model3.head())
    params_df_model3.to_csv('bart_wallsten.csv')


    # === Wullhorst Model 1 (for hot CCT) ===
    hot = HOTModel(hot_df)

    def fit_one_user(uid, user_flips):
        try:
            rho_est, lam_est, beta_est = hot.fit(user_flips)
            return {
                'user_id': uid,
                'wullhorst1_hot_rho': rho_est,
                'wullhorst1_hot_lambda': lam_est,
                'wullhorst1_hot_beta': beta_est
            }
        except Exception as e:
            print(f"Error with user {uid}: {e}")
            return {
                'user_id': uid,
                'wullhorst1_hot_rho': np.nan,
                'wullhorst1_hot_lambda': np.nan,
                'wullhorst1_hot_beta': np.nan
            }

    results = []
    n_jobs = 8 

    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = {
            executor.submit(fit_one_user, uid, user_flips): uid
            for uid, user_flips in hot_df.groupby('user_id', sort=False)
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Parallel Fitting"):
            results.append(future.result())

    cct_hot_params_df = pd.DataFrame(results)
    print(cct_hot_params_df.head())
    cct_hot_params_df.to_csv('hot_wullhorst1.csv')

    # === Wullhorst Model 2 (for hot CCT) ===
    hot2 = HOTModel2(hot_df)
    results = []

    def fit_one_user(uid, user_flips):
        try:
            rho, lam, delta, eta, beta = hot2.fit(user_flips, n_starts=30, n_jobs=4)
            return {"user_id": uid, "hot_wullhorst2_rho": rho, "hot_wullhorst2_lambda": lam,
                    "hot_wullhorst2_delta": delta, "hot_wullhorst2_eta": eta, "hot_wullhorst2_beta": beta}
        except Exception as e:
            print(f"Error with user {uid}: {e}")
            return {"user_id": uid, "hot_wullhorst2_rho": np.nan, "hot_wullhorst2_lambda": np.nan,
                    "hot_wullhorst2_delta": np.nan, "hot_wullhorst2_eta": np.nan, "hot_wullhorst2_beta": np.nan}

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(fit_one_user, uid, df) for uid, df in hot_df.groupby("user_id")]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Users"):
            results.append(f.result())

    model2_params_df = pd.DataFrame(results)
    print(model2_params_df.head())
    model2_params_df.to_csv('hot_wullhorst2.csv')

    # === Haffke Model (for hot CCT) ===
    haffke_df = prepare_haffke_input(hot_df)
    results = []
    for uid, user_flips in tqdm(haffke_df.groupby('user_id', sort=False), desc="Fitting Haffke hot (per user)"):
        user_model = HaffkeModel(user_flips)
        try:
            params = user_model.fit()         
        except RuntimeError as e:
            print(f"Fit failed for user {uid}: {e}")
            params = np.array([np.nan, np.nan, np.nan, np.nan])

        results.append({
            'user_id': uid,
            'hot_haffke_alpha': params[0],
            'hot_haffke_delta': params[1],
            'hot_haffke_eta': params[2],
            'hot_haffke_theta': params[3]
        })

    haffke_hot_params_df = pd.DataFrame(results)
    print(haffke_hot_params_df.head())
    haffke_hot_params_df.to_csv('hot_haffke.csv')

    # === Wullhorst Model 1 (for cold CCT) ===
    cold = COLDModel(cold_df)
    user_ids = cold_df['user_id'].unique()
    results = []
    for user_id in tqdm(user_ids, desc="Fitting Prospect Theory models"):
        user_data = cold_df[cold_df['user_id'] == user_id]
        params = cold.fit(user_data)
        params['user_id'] = user_id
        results.append(params)
    cct_cold_params_df = pd.DataFrame(results)
    print('CCT-cold params df')
    print(cct_cold_params_df.head())
    cct_cold_params_df.to_csv('cold_wullhorst1.csv')

    # === Wullhorst Model 2 (for cold CCT) ===
    cold2 = COLDModel2(cold_df)
    results = []
    for uid, df_user in tqdm(cold_df.groupby("user_id", sort=False), desc="Fitting COLD Model2 per user"):
        try:
            rho, lam, delta, eta, beta = cold2.fit(df_user, n_starts=30, n_jobs=4)
            results.append({
                "user_id": uid, "rho": rho, "lambda": lam,
                "delta": delta, "eta": eta, "beta": beta
            })
        except Exception as e:
            print(f"Error with user {uid}: {e}")
            results.append({"user_id": uid, "cold_wullhorst2_rho": np.nan, "cold_wullhorst2_lambda": np.nan,
                            "cold_wullhorst2_delta": np.nan, "cold_wullhorst2_eta": np.nan, "cold_wullhorst2_beta": np.nan})

    params_df = pd.DataFrame(results)
    print(params_df.head())
    params_df.to_csv('cold_wullhorst2.csv')

    # === Haffke Model (for cold CCT) ===
    results = []
    for uid in tqdm(cold_df['user_id'].unique(), desc="Fitting Haffke cold"):
        user_data = cold_df[cold_df['user_id'] == uid]
        haffke_cold_user = HaffkeColdModel_Model3(user_data)  # новый объект на пользователя
        res = haffke_cold_user.fit(n_starts=20, random_seed=uid)
        results.append({
            'user_id': uid,
            'cold_haffke_alpha': res.x[0],
            'cold_haffke_lambda': res.x[1],
            'cold_haffke_delta': res.x[2],
            'cold_haffke_eta': res.x[3],
            'cold_haffke_theta': res.x[4]
        })

    haffke_cold_params_df = pd.DataFrame(results)
    print('Haffke-cold params df')
    print(haffke_cold_params_df.head())
    haffke_cold_params_df.to_csv('cold_haffke.csv')