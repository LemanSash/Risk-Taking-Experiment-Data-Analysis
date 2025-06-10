import numpy as np
from scipy.stats import pearsonr, spearmanr

class QuestionnaireAnalyzer:
    """
    Анализ опросника: очистка, инверсия и расчет альфа Кронбаха.
    """
    def __init__(self, users_df, responses_df):
        self.users = users_df
        self.responses = responses_df
        self.reverse_questions = [4,5,13,14,15,16,17,19,20,21,26]

    def preprocess(self):
        # Оставим только пользователей из users
        dupes = self.responses.duplicated(subset=['user_id', 'question_number'], keep=False)
        self.responses[dupes].sort_values(by=['user_id', 'question_number'])
        unique_responses = self.responses.drop_duplicates(subset=['user_id', 'question_number'], keep='first')

        # Фильтрация по пользователям из users
        filtered_responses = unique_responses[unique_responses['user_id'].isin(self.users['user_id'])]

        # Pivot: строки — пользователи, столбцы — номера вопросов, значения — ответы
        response_matrix = filtered_responses.pivot(index='user_id', columns='question_number', values='response')

        # Удалим пользователей с пропущенными ответами
        response_matrix_clean = response_matrix.dropna(axis=0)

        # Инвертируем ответы на обратные вопросы (предполагаем шкалу 1–5)
        for q in self.reverse_questions:
            if q in response_matrix_clean.columns:
                response_matrix_clean[q] = 6 - response_matrix_clean[q]

        return self.response_matrix_clean

    @staticmethod
    def cronbach_alpha(df):
        k = df.shape[1]
        var_items = df.var(axis=0, ddof=1)
        var_total = df.sum(axis=1).var(ddof=1)
        return (k/(k-1))*(1 - var_items.sum()/var_total)

    def compute_alpha(self):
        df_clean = self.preprocess()
        return self.cronbach_alpha(df_clean)
    
    def count_questions(self, user_id: int):
        if user_id not in self.responses['user_id'].values:
            return np.nan
        total = 0
        user_quest = self.responses[
            self.responses['user_id'] == user_id
        ].drop_duplicates('question_number').head(30)
        numbers = list(user_quest['question_number'])
        answers = list(user_quest['response'])
        for index, number in enumerate(numbers):
            if number in self.reverse_questions:
                total += (6 - answers[index])
            else:
                total += answers[index]
        return total
    
    def transfrom_scale(df):
        df_analysis = df.copy()
        MIN_SCORE = 30
        MAX_SCORE = 150
        RANGE = MAX_SCORE - MIN_SCORE

        df_analysis['questionnaire_scaled'] = ((df_analysis['questionnaire'] - MIN_SCORE) / RANGE) * 100
        df_analysis['questionnaire_scaled'] = df_analysis['questionnaire_scaled'].round(2)
        return df_analysis

class TaskAnalyzer:
    """
    Подсчет классических метрик для BART, CCT и IGT.
    """
    def __init__(self, users_df, bart_df, cct_hot_df, cct_cold_df, igt_df):
        self.users = users_df
        self.bart = bart_df
        self.cct_hot = cct_hot_df
        self.cct_cold = cct_cold_df
        self.igt = igt_df

    def count_bart(self, user_id: int):
        if user_id not in self.bart['user_id'].values:
            return np.nan, np.nan
        user_pumps = self.bart[self.bart['user_id'] == user_id]
        user_pumps = user_pumps[user_pumps['popped'] == 0]
        pumps = np.mean(user_pumps['pumps'])
        rts = np.mean(user_pumps['reaction_time'])
        return pumps, rts

    def count_cct_hot(self, user_id: int):
        if user_id not in self.cct_hot['user_id'].values:
            return np.nan
        user_cards = self.cct_hot[self.cct_hot['user_id'] == user_id]
        user_cards = user_cards[user_cards['trial_type'] == 'experimental']
        cards = np.mean(user_cards['flipped_cards'])
        return cards

    def count_cct_cold(self, user_id: int):
        if user_id not in self.cct_cold['user_id'].values:
            return np.nan, np.nan
        user_cards = self.cct_cold[self.cct_cold['user_id'] == user_id]
        cards = np.mean(user_cards['num_cards'])
        rts = np.mean(user_cards['reaction_time'])
        return cards, rts

    def count_igt(self, user_id: int):
        if user_id not in self.igt['user_id'].values:
            return np.nan, np.nan
        user_cards = self.igt[self.igt['user_id'] == user_id]
        user_cards = user_cards[user_cards['trial_number'] > 40]
        a_cards = user_cards[user_cards['deck'] == 'A'].count()
        b_cards = user_cards[user_cards['deck'] == 'B'].count()
        c_cards = user_cards[user_cards['deck'] == 'C'].count()
        d_cards = user_cards[user_cards['deck'] == 'D'].count()
        a_and_b = a_cards['result_id'] + b_cards['result_id']
        c_and_d = c_cards['result_id'] + d_cards['result_id']
        rts = np.mean(user_cards['reaction_time'])
        net_score = c_and_d - a_and_b
        return net_score, rts