import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class LinearModel:
    def __init__(self, data_df):
        self.data = data_df.copy()
        self.X = None
        self.y = None

    def construct_model(self):
        df = self.data

        # Разделение на X и y
        self.X = df.drop(columns=['user_id', 'questionnaire'])
        self.y = df['questionnaire']
        self.X = sm.add_constant(self.X)

        # Обучение модели
        model = sm.OLS(self.y, self.X).fit()
        return model
    
    def get_metrics(self, model, method):
        if method == 'summary':
            return model.summary()
        elif method == 'r2':
            return model.rsquared
        elif method == 'f':
            return (model.fvalue, model.f_pvalue)
        else:
            print('No such method')
            return np.nan

    def visualise_model(self, model):
        y_pred = model.predict(self.X)
        residuals = self.y - y_pred

        plt.figure(figsize=(8,6))
        sns.scatterplot(x=self.y, y=y_pred, color='royalblue', edgecolor='black', s=60, alpha=0.6)
        plt.xlabel('Фактические questionnaire')
        plt.ylabel('Предсказанные')
        plt.title('Модель 1: Фактические vs Предсказанные')
        plt.plot([self.y.min(), self.y.max()], [self.y.min(), self.y.max()], color='red', linestyle='--')
        plt.show()

        plt.figure(figsize=(8,6))
        sns.scatterplot(x=y_pred, y=residuals)
        plt.axhline(0, color='r', linestyle='--')
        plt.xlabel('Предсказанные значения')
        plt.ylabel('Остатки')
        plt.title('Модель 1: Остатки vs Предсказанные')
        plt.show()

    @staticmethod
    def backward_elimination(X, y, significance_level=0.05):
        X_const = sm.add_constant(X)
        features = list(X_const.columns)

        while True:
            model = sm.OLS(y, X_const[features]).fit()
            p_vals = model.pvalues.drop('const', errors='ignore')

            max_p = p_vals.max()
            if max_p > significance_level:
                worst = p_vals.idxmax()
                features.remove(worst)
                print(f'Исключаем признак: {worst} (p-value={max_p:.4f})')
            else:
                break

        final_model = sm.OLS(y, X_const[features]).fit()
        return final_model, X_const[features]
    
    def search_params(self):
        X = self.data.drop(columns=['user_id', 'questionnaire'])
        y = self.data['questionnaire']
        model, X_selected = self.backward_elimination(X, y)
        return model, X_selected
