import pandas as pd
import numpy as np
import random
from scipy import stats
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from data_loader import DataLoader
from analyzers import QuestionnaireAnalyzer, TaskAnalyzer
from models.vse_model import VSEModel
from models.stld_model import STLDModel
from models.pospect_model_hot import HOTModel
from models.prospect_model_cold import COLDModel
from utils import DataPreprocessor
from models.regression_model import LinearModel

if __name__ == '__main__':
    DATA_DIR = 'C:/Users/alina/python_projects/experiment_data/data'

    # Загрузка данных
    loader = DataLoader(DATA_DIR).load_all()
    ids = loader.users['user_id']

    # Анализ опросника
    q_analyzer = QuestionnaireAnalyzer(loader.users, loader.q_responses)
    alpha = q_analyzer.compute_alpha()
    print(f"Альфа Кронбаха: {alpha:.4f}")

    # Подсчет метрик задач
    task_an = TaskAnalyzer(loader.users, loader.bart, loader.cct_hot, loader.cct_cold, loader.igt)

    # Считаем метрики BART
    b_results = []
    b_rts = []
    for id in ids:
        pumps, rts = task_an.count_bart(id)
        b_results.append(pumps)
        b_rts.append(rts)
    loader.users['bart_results'] = b_results
    loader.users['bart_rts'] = b_rts

    # Считаем метрики IGT
    igt_results = []
    igt_rts = []
    for id in ids:
        pumps, rts = task_an.count_igt(id)
        igt_results.append(pumps)
        igt_rts.append(rts)
    loader.users['igt_results'] = b_results
    loader.users['igt_rts'] = b_rts

    # Считаем метрики CCT-hot
    hot_results = []
    for id in ids:
        hot_results.append(task_an.count_cct_hot(id))
    loader.users['cct_hot_results'] = hot_results

    # Считаем метрики CCT-cold
    cold_results = []
    cold_rts = []
    for id in ids:
        cold_result, cold_rt = task_an.count_cct_cold(id)
        cold_results.append(cold_result)
        cold_rts.append(cold_rt)
    loader.users['cct_cold_results'] = cold_results
    loader.users['cct_cold_rts'] = cold_rts

    # Считаем баллы опросника
    quest_results = []
    for id in ids:
        quest_results.append(q_analyzer.count_questions(id))
    loader.users['questionnaire'] = quest_results

    loader.users = q_analyzer.transfrom_scale(loader.users)

    # Делаем попарные датасеты и проводим корреляционный анализ пов. методик
    preprocess = DataPreprocessor()
    pair_datasets = preprocess.pair_datasets(loader.users)

    for pair_name, df in pair_datasets.items():
        print(f"=== Анализ корреляций для {pair_name} ===")

        # Выделяем все колонки с результатами, кроме user_id и education
        cols = [col for col in df.columns if col not in ['user_id', 'education']]
        
        # Анализ с порогом корреляции
        corr_matrix = preprocess.analyze_correlations(df, pair_name, cols, threshold=0.6)
        print('Матрица корреляций для поведенческих методик')
        print(corr_matrix)

    # Проводим корреляционный анализ пов. методик и опросника
    method_cols = ['bart_results', 'cct_hot_results', 'cct_cold_results', 'igt_result']
    for method in method_cols:
        preprocess.corr_method_questionnaire(loader.users.dropna(), method)

    # Создаём датасет с участниками, прошедшими все методики
    complete_users = preprocess.all_methods(loader.users)
    complete_users = preprocess.normalize_metrics(complete_users)

    #Готовим данные для моделей
    valid_user_ids = set(complete_users['user_id'].unique())
    igt_df = loader.igt[loader.igt['user_id'].isin(valid_user_ids)]
    bart_df = loader.bart[loader.bart['user_id'].isin(valid_user_ids)]
    hot_df = loader.cct_hot[loader.cct_hot['user_id'].isin(valid_user_ids) & (loader.cct_hot['trial_type'] == 'experimental')]
    cold_df = loader.cct_cold[loader.cct_cold['user_id'].isin(valid_user_ids)]
    random_user = random.choice(valid_user_ids)

    # Подгонка модели VSE
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
                'igt_alpha_pos': params[0],
                'igt_alpha_neg': params[1],
                'igt_beta': params[2],
                'igt_gamma': params[3],
            })
        except Exception as e:
            print(f"Ошибка при обработке user_id={uid}: {e}")

    igt_params_df = pd.DataFrame(vse_results)

    # PPC для VSE
    user_df = igt_df[igt_df['user_id']==random_user].sort_values('trial_number')
    best_params = vse.fit(user_df)
    vse_ppc_r2 = vse.predictive_check(user_df)

    # Parameter Recovery для IGT
    vse_recovery_df = vse.parameter_recovery(n_subjects=50, n_trials=200)

    # Модель Scaled Target Learning with Decay (BART)
    stld = STLDModel(bart_df)
    stld_params_list = []

    # Перебираем только пользователей, прошедших все методики
    for user_id in tqdm(bart_df, desc="Fitting STL-D models"):
        user_df = bart_df[bart_df['user_id'] == user_id]
        if len(user_df) < 5:
            continue
        try:
            params = stld.fit(user_df)
            stld_params_list.append({
                'user_id': user_id,
                'bart_stld_w1': params[0],
                'bart_stld_vwin': params[1],
                'bart_stld_vloss': params[2],
                'bart_stld_alpha': params[3],
                'bart_stld_beta': params[4]
            })
        except Exception as e:
            print(f"Ошибка для пользователя {user_id}: {e}")
    bart_params_df = pd.DataFrame(stld_params_list)

    # PPC для STL-D
    user_df = bart_df[bart_df['user_id']==random_user]
    stld_ppc_r2 = stld.predictive_check(user_df)

    # Parameter Recovery дл STL-D
    stld_recovery_df = stld.parameter_recovery(n_subjects=50, n_trials=1000)

    # Модель Prospect Theory + Expected Utility (CCT-hot)
    hot_df = preprocess.expand_to_flips(hot_df)
    hot = HOTModel(hot_df)
    
    results = []
    for uid, user_flips in hot_df.groupby('user_id', sort=False):
        rho_est, lam_est, beta_est = hot.fit(user_flips)
        results.append({
            'user_id': uid,
            'hot_rho': rho_est,
            'hot_lambda': lam_est,
            'hot_beta': beta_est
        })

    cct_hot_params_df = pd.DataFrame(results)

    # PPC для HOT Model
    user_df = hot_df[hot_df['user_id']==random_user]
    hot_ppc_r2 = hot.predictive_check(user_df)

    # Parameter Recovery для HOT Model
    uids = hot_df['user_id'].unique()[:3]
    template = pd.concat([hot_df[hot_df['user_id'] == uid] for uid in uids])
    template['trial_number'] = template.groupby(['user_id', 'trial_number']).ngroup()
    recovery_df = hot.parameter_recovery(template, n_subjects=50)

    # Модель Prospect Theory + Expected Utility (CCT-cold)
    cold = COLDModel(cold_df)
    user_ids = cold_df['user_id'].unique()
    # Сбор параметров
    results = []
    for user_id in tqdm(user_ids, desc="Fitting Prospect Theory models"):
        user_data = cold_df[cold_df['user_id'] == user_id]
        params = cold.fit(user_data)
        params['user_id'] = user_id
        results.append(params)
    cct_cold_params_df = pd.DataFrame(results)

    # PPC для COLD Model
    user_data = cold_df[cold_df['user_id'] == random_user]
    cold_ppc_r2 = cold.predictive_check(user_data, N=32)

    # Parameter Recovery для COLD Model
    template_data = cold_df[cold_df['user_id'] == random_user].reset_index(drop=True)
    recovery_df = cold.parameter_recovery(template_data, n_subjects=50, N=32)

    # Собираем итоговый датафрейм со всеми параметрами
    results_df = pd.DataFrame()
    results_df = pd.merge(igt_params_df, bart_params_df, on="user_id", how='inner')
    results_df = pd.merge(results_df, cct_hot_params_df, on="user_id", how='inner')
    results_df = pd.merge(results_df, cct_cold_params_df, on="user_id", how='inner')
    results_df.to_excel('all_methods_params.xlsx')

    # Матрица корреляций
    corr_matrix = results_df.corr()
    print("Корреляционная матрица:")
    print(corr_matrix)

    # Обработка итогового датасета
    results_df.dropna()

    z_scores = np.abs(stats.zscore(results_df))
    outliers = (z_scores > 3)
    print(f"Число выбросов (|z|>3): {np.sum(outliers)}")
    results_df = results_df.clip(
        lower=results_df.mean() - 3*results_df.std(),
        upper=results_df.mean() + 3*results_df.std(),
        axis=1
    )

    # Проверка скошенности и преобразования
    skewed = results_df.skew().abs() > 1
    print("Сильно скошенные колонки:", results_df.columns[skewed].tolist())

    # Лог-преобразование (для положительных параметров)
    for col in results_df.columns[skewed]:
        shift = 1 - results_df[col].min() if results_df[col].min() <= 0 else 0
        results_df[col] = np.log(results_df[col] + shift)

    # Стандартизация
    scaler = StandardScaler()
    results_scaled = pd.DataFrame(
        scaler.fit_transform(results_df),
        columns=results_df.columns,
        index=results_df.index
    )

    # Строим линейную регрессию для параметров VSE и опросника
    igt_columns = igt_params_df.columns()
    igt_linear_df = results_scaled[igt_columns].merge(loader.users[['user_id', 'questionnaire_scaled']], on='user_id')
    igt_linear = LinearModel(igt_linear_df)
    igt_model = igt_linear.construct_model()
    print(igt_linear.get_metrics(igt_model, 'summary'))
    igt_linear.visualise_model(igt_model)

    # Строим линейную регрессию для параметров STL-D и опросника
    bart_columns = bart_params_df.columns()
    bart_linear_df = results_scaled[bart_columns].merge(loader.users[['user_id', 'questionnaire_scaled']], on='user_id')
    bart_linear = LinearModel(bart_linear_df)
    bart_model = bart_linear.construct_model()
    print(bart_linear.get_metrics(bart_model, 'summary'))
    bart_linear.visualise_model(bart_model)

    # Строим линейную регрессию для параметров HOT Model и опросника
    cct_hot_columns = cct_hot_params_df.columns()
    hot_linear_df = results_scaled[cct_hot_columns].merge(loader.users[['user_id', 'questionnaire_scaled']], on='user_id')
    hot_linear = LinearModel(hot_linear_df)
    hot_model = hot_linear.construct_model()
    print(hot_linear.get_metrics(hot_model, 'summary'))
    hot_linear.visualise_model(hot_model)

    # Строим линейную регрессию для параметров COLD Model и опросника
    cct_cold_columns = cct_cold_params_df.columns()
    cold_linear_df = results_scaled[cct_cold_columns].merge(loader.users[['user_id', 'questionnaire_scaled']], on='user_id')
    cold_linear = LinearModel(cold_linear_df)
    cold_model = cold_linear.construct_model()
    print(cold_linear.get_metrics(cold_model, 'summary'))
    cold_linear.visualise_model(cold_model)

    # Линейная регрессия с параметрами всех моделей и опросника
    all_df = results_scaled.merge(loader.users[['user_id', 'questionnaire_scaled']], on='user_id').dropna()
    all_linear = LinearModel(all_df)
    all_model, params = all_linear.search_params()
    print(all_linear.get_metrics(all_model, 'summary'))
    all_linear.visualise_model(all_model)