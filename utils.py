import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler, PowerTransformer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from IPython.display import display

class DataPreprocessor:
    @staticmethod
    def encode_categorical(df):
        df_encoded = df.copy()
        gender_mapping = {'female': 0, 'male': 1}

        if 'gender' in df_encoded.columns:
            df_encoded['gender'] = df_encoded['gender'].map(gender_mapping)

        if 'education' in df_encoded.columns:
            df_encoded['education'] = df_encoded['education'].fillna('unknown').astype(str)
            encoder = OneHotEncoder(sparse_output=False, drop='first')
            encoded = encoder.fit_transform(df_encoded[['education']])
            new_cols = ['edu_' + cat for cat in encoder.categories_[0][1:]]
            df_encoded[new_cols] = encoded

        return df_encoded

    @staticmethod
    def treat_outliers(df, column, method='winsorize'):
        df_clean = df.copy()
        q1 = df_clean[column].quantile(0.25)
        q3 = df_clean[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        if method == 'winsorize':
            df_clean[column] = np.clip(df_clean[column], lower, upper)
        elif method == 'remove':
            df_clean = df_clean[(df_clean[column] >= lower) & (df_clean[column] <= upper)]

        return df_clean
    
    @staticmethod
    def check_normality(df, columns=None, alpha=0.05):
        """
        Проверка нормальности распределения для выбранных колонок.
        
        columns : list или None
            Список колонок для проверки. Если None, то используется
            ['bart_results', 'cct_hot_results', 'cct_cold_results', 'igt_results', 'questionnaire_scaled'].
        alpha : float
            Уровень значимости для статистических тестов (по умолчанию 0.05).
        """

        if columns is None:
            columns = ['bart_results', 'cct_hot_results', 'cct_cold_results', 'igt_results', 'questionnaire_scaled']

        for col in columns:
            if col not in df.columns:
                print(f"Колонка {col} отсутствует в DataFrame")
                continue

            data = df[col].dropna()

            print(f"\nПроверка для колонки: {col} (n={len(data)})")

            # Тест Шапиро-Уилка
            shapiro_stat, shapiro_p = stats.shapiro(data)
            print(f"   Shapiro-Wilk: stat={shapiro_stat:.3f}, p={shapiro_p:.3f} "
                f"-> {'Нормальное' if shapiro_p > alpha else 'Не нормальное'}")

            # Тест Д’Агостино-Пирсона
            dag_stat, dag_p = stats.normaltest(data)
            print(f"   D’Agostino-Pearson: stat={dag_stat:.3f}, p={dag_p:.3f} "
                f"-> {'Нормальное' if dag_p > alpha else 'Не нормальное'}")

            # Визуализация
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))

            sns.histplot(data, kde=True, ax=axes[0])
            axes[0].set_title(f"Гистограмма: {col}")

            stats.probplot(data, dist="norm", plot=axes[1])
            axes[1].set_title(f"QQ-plot: {col}")

            plt.tight_layout()
            plt.show()
    
    @staticmethod
    def normalize_metrics(df, column, norm_method='z', plot=False):
        df = df.copy()
        if column not in df.columns:
            raise ValueError(f"Колонка '{column}' отсутствует в DataFrame")
        original = df[column].copy()

        if norm_method == 'robust':
            scaler = RobustScaler()
            df[column] = scaler.fit_transform(df[column])
            prefix = 'robust_'
        elif norm_method == 'minmax':
            scaler = MinMaxScaler()
            df[column] = scaler.fit_transform(df[column])
            prefix = 'minmax_'
        elif norm_method == 'maxabs':
            scaler = MaxAbsScaler()
            df[column] = scaler.fit_transform(df[column])
            prefix = 'maxabs_'
        elif norm_method == 'log':
            shift = 1 - df[column].min() if (df[column] <= 0).any() else 0
            df[column] = np.log(df[column] + shift)
            prefix = 'log_'
        elif norm_method == 'yeo':
            scaler = PowerTransformer(method='yeo-johnson')
            df[column] = scaler.fit_transform(df[[column]])
            prefix = 'yeo_'
        else:
            df[column] = (df[column] - df[column].mean()) / df[column].std()
            prefix = 'z_'

        #df.rename(columns={col: prefix + col for col in metric_cols}, inplace=True)
        new_col = prefix + column
        df.rename(columns={column: new_col}, inplace=True)
        
        if plot:
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            sns.histplot(original, kde=True)
            plt.title(f'До: {column}')
            plt.subplot(1, 2, 2)
            sns.histplot(df[new_col], kde=True)
            plt.title(f'После: {new_col}')
            plt.tight_layout()
            plt.show()

        return df

    @staticmethod
    def compare_normalization(df, column: str):
        """Сравнение методов нормализации для одной колонки"""
        original = df[column].copy()
        methods = ['robust', 'minmax', 'maxabs', 'z', 'log']
        results = {}
        
        plt.figure(figsize=(15, 4 * len(methods)))
        
        for i, method in enumerate(methods, 1):
            temp_df = pd.DataFrame({column: original})
            temp_df = DataPreprocessor.normalize_metrics(temp_df, norm_method=method, plot=False)
            new_col = f"{'robust' if method=='robust' else 'z' if method=='z' else method}_{column}"
            results[method] = temp_df[new_col]
            
            # Гистограмма
            plt.subplot(len(methods), 2, 2*i-1)
            sns.histplot(results[method], kde=True)
            plt.title(f"{method} (skew: {results[method].skew():.2f})")
            
            # Boxplot
            plt.subplot(len(methods), 2, 2*i)
            sns.boxplot(x=results[method])
            plt.title(f"{method} (IQR: {results[method].quantile(0.75)-results[method].quantile(0.25):.2f})")
        
        plt.tight_layout()
        plt.show()

        metrics = {
            "skewness": lambda x: x.skew(),
            "kurtosis": lambda x: x.kurtosis(),
            "shapiro-wilk": lambda x: stats.shapiro(x)[0],
            "anderson-darling": lambda x: stats.anderson(x, dist='norm').statistic
        }

        for method, data in results.items():
            print(f"\nМетод: {method}")
            for name, func in metrics.items():
                print(f"{name}: {func(data):.4f}")
        
        return results
    
    @staticmethod
    def pair_datasets(users_df):
        pairs = {
            'bart_cct_cold': ['bart_results', 'cct_cold_results'],
            'bart_cct_hot': ['bart_results', 'cct_hot_results'],
            'bart_igt': ['bart_results', 'igt_results'],
            'cct_cold_hot': ['cct_cold_results', 'cct_hot_results'],
            'cct_cold_igt': ['cct_cold_results', 'igt_results'],
            'cct_hot_igt': ['cct_hot_results', 'igt_results'],
        }

        processed_dfs = {}

        for name, cols in pairs.items():
            df = users_df[['user_id'] + cols].dropna()
            # Добавим gender и education, если есть
            for extra in ['gender', 'education']:
                if extra in users_df.columns:
                    df[extra] = users_df.set_index('user_id').loc[df['user_id']][extra].values

            # Предобработка
            df = DataPreprocessor.encode_categorical(df)
            for col in cols:
                df = DataPreprocessor.treat_outliers(df, col)
            print(name, len(df))
            processed_dfs[name] = df
        return processed_dfs
    
    @staticmethod
    def analyze_correlations(df, pair_name, cols, threshold=0.6):
        """
        Анализ корреляций с расчетом p-value.
        """
        missing_cols = [col for col in cols if col not in df.columns]
        if missing_cols:
            print(f"В датафрейме отсутствуют колонки: {missing_cols}")
            return None
        
        corr_df = df[cols]

        if corr_df.shape[1] < 2:
            print(f"В паре {pair_name} недостаточно колонок для корреляционного анализа.")
            return None

        # Корреляционная матрица
        corr_matrix = corr_df.corr(method='spearman')

        # Тепловая карта
        plt.figure(figsize=(8, 6))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm',
                    center=0, vmin=-1, vmax=1, linewidths=0.5)
        plt.title(f'Корреляции методов ({pair_name})', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

        # Поиск сильных корреляций
        strong_corrs = []
        all_corrs = []
        n = len(cols)

        for i in range(n):
            for j in range(i + 1, n):
                col1, col2 = cols[i], cols[j]
                #pearson_r, pearson_p = stats.pearsonr(df[col1], df[col2])
                spearman_r, spearman_p = stats.spearmanr(df[col1], df[col2])
                kendall_r, kendall_p = stats.kendalltau(df[col1], df[col2])

                all_corrs.append({
                    'pair': f"{col1} - {col2}",
                    'kendall_r': round(kendall_r, 3),
                    'kendall_p': round(kendall_p, 4),
                    'spearman_r': round(spearman_r, 3),
                    'spearman_p': round(spearman_p, 4)
                })

                if abs(spearman_r) > threshold:
                    strong_corrs.append({
                        'pair': f"{col1} - {col2}",
                        'kendall': round(kendall_r, 3),
                        'kendall_p': round(kendall_p, 4),
                        'spearman': round(spearman_r, 3),
                        'spearman_p': round(spearman_p, 4),
                        'interpretation': "Сильная положительная" if spearman_r > 0 else "Сильная отрицательная"
                    })

        # Вывод всех корреляций с p-value
        print("\nВсе рассчитанные корреляции и p-value:")
        display(pd.DataFrame(all_corrs))

        # Вывод сильных корреляций
        if strong_corrs:
            print(f"\nСильные корреляции (|r| > {threshold}) в паре {pair_name}:")
            display(pd.DataFrame(strong_corrs))

            plt.figure(figsize=(12, 4 * len(strong_corrs)))
            for idx, corr in enumerate(strong_corrs, 1):
                col1, col2 = corr['pair'].split(' - ')
                plt.subplot(len(strong_corrs), 1, idx)
                sns.regplot(x=df[col1], y=df[col2], scatter_kws={'alpha': 0.5})
                plt.title(
                    f"{corr['pair']}\n"
                    f"Kendall: {corr['kendall']} (p={corr['kendall_p']}) | "
                    f"Spearman: {corr['spearman']} (p={corr['spearman_p']})"
                )
                plt.xlabel(col1)
                plt.ylabel(col2)

            plt.tight_layout()
            plt.show()
        else:
            print(f"\nВ паре {pair_name} нет сильных корреляций (|r| > {threshold})")

        return corr_matrix
    
    @staticmethod
    def corr_method_questionnaire(df, method_col, questionnaire_col='questionnaire_scaled'):
        # Pearson
        pearson_r, pearson_p = stats.pearsonr(df[method_col], df[questionnaire_col])
        # Spearman
        spearman_r, spearman_p = stats.spearmanr(df[method_col], df[questionnaire_col])
        
        print(f"Корреляция {method_col.replace('minmax_', '')} и опросника:")
        print(f"  Pearson  r = {pearson_r:.3f}, p = {pearson_p:.4f}")
        print(f"  Spearman r = {spearman_r:.3f}, p = {spearman_p:.4f}")
        print('-'*40)

    @staticmethod
    def all_methods(users_df):
        METHOD_COLUMNS = {
            'IGT': ['igt_results', 'igt_rts'],  # Столбцы первой методики
            'BART': ['bart_results', 'bart_rts'],  # Столбцы второй методики
            'CCT-hot': ['cct_hot_results'],                  # Столбцы третьей методики
            'CCT-cold': ['cct_cold_results', 'cct_cold_rts']  # Столбцы четвертой методики
        }
        
        complete_users_mask = pd.Series(True, index=users_df.index)

        for method, columns in METHOD_COLUMNS.items():
            method_mask = users_df[columns].notnull().all(axis=1)
            complete_users_mask &= method_mask

        complete_users = users_df[complete_users_mask].copy()

        print(f"Всего пользователей: {len(users_df)}")
        print(f"Пользователей со всеми методиками: {len(complete_users)}")
        print(f"Доля полных данных: {len(complete_users)/len(users_df):.1%}")

        missing_data = pd.DataFrame(index=users_df.index)

        for method, columns in METHOD_COLUMNS.items():
            missing_data[method] = users_df[columns].isnull().any(axis=1)

        print("\nКаких данных не хватает:")
        print(missing_data[~complete_users_mask].sum())
        return complete_users
    
    @staticmethod
    def encode_igt_decks(igt_df):
        deck_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        igt_df['deck_num'] = igt_df['deck'].map(deck_mapping)
        return igt_df
    
    @staticmethod
    def expand_to_flips(cct_df):
        records = []
        df_exp = cct_df[cct_df['trial_type'] == 'experimental']
        
        for _, row in df_exp.iterrows():
            uid        = row['user_id']
            tnum       = int(row['trial_number'])
            n_flips    = int(row['flipped_cards'])
            gain_amt   = float(row['gain_amount'])
            loss_amt   = float(row['loss_amount'])
            points     = float(row['points'])
            bad_cards = int(row['loss_cards'])

            # Ожидаемое количество очков, если все флипы — выигрышные
            expected_points = n_flips * gain_amt

            # True, если последний флип был проигрышным
            popped = points < expected_points

            for k in range(1, n_flips + 1):
                if popped and k == n_flips:
                    outcome = -loss_amt
                else:
                    outcome = +gain_amt

                records.append({
                    'user_id': uid,
                    'trial_number': tnum,
                    'flip_number': k,
                    'outcome': outcome,
                    'flips': n_flips,
                    'gain_amount': gain_amt,
                    'loss_amount': loss_amt,
                    'popped': popped,
                    'loss_cards': bad_cards
                })

        flips_df = pd.DataFrame.from_records(records)
        flips_df.sort_values(['user_id', 'trial_number', 'flip_number'], inplace=True)
        flips_df.reset_index(drop=True, inplace=True)
        return flips_df
    
    @staticmethod
    def add_choice_column(df):
        df = df.copy()
        df['choice'] = 1  # по умолчанию продолжил

        for (_, _), trial in df.groupby(['user_id', 'trial_number'], sort=False):
            last_idx = trial.index[-1]
            popped = trial['popped'].iloc[-1]
            if not popped:
                # остановился сам -> в последней строке решение = stop
                df.at[last_idx, 'choice'] = 0
            else:
                # взорвался -> последняя карта взята по решению continue (оставляем 1)
                pass
        return df



    @staticmethod
    def get_users_info(users):
        mean_age = np.mean(users['age'])
        std_age = np.std(users['age'], ddof=1)  # выборочное стандартное отклонение
        
        total = len(users)
        
        males_count = users[users['gender'] == 'male'].shape[0]
        females_count = users[users['gender'] == 'female'].shape[0]
        
        if males_count > 0:
            males_pct = males_count / total * 100
        else:
            males_pct = 0
        females_pct = females_count / total * 100
        
        high_school_count = users[users['education'] == 'high_school'].shape[0]
        bachelor_count = users[users['education'] == 'bachelor'].shape[0]
        master_count = users[users['education'] == 'master'].shape[0]
        other_count = users[users['education'] == 'other'].shape[0]
        
        # Формируем итоговую строку
        result = (
            f"Средний возраст участников составил {mean_age:.1f} (SD = {std_age:.1f}) лет.\n"
            f"Половой состав: {males_count} мужчин ({males_pct:.1f}%) и {females_count} женщин ({females_pct:.1f}%).\n"
            f"Уровень образования: {high_school_count} с аттестатом о среднем образовании ({high_school_count / total * 100:.1f}%), "
            f"{bachelor_count} со степенью бакалавра ({bachelor_count / total * 100:.1f}%), "
            f"{master_count} со степенью магистра ({master_count / total * 100:.1f}%), "
            f"и {other_count} с иным образованием ({other_count / total * 100:.1f}%)."
        )
        
        return result

