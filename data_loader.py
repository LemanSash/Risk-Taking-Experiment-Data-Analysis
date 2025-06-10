import pandas as pd

class DataLoader:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.users = None
        self.q_responses = None
        self.bart = None
        self.cct_cold = None
        self.cct_hot = None
        self.igt = None

    def load_all(self):
        self.load_users()
        self.load_questionnaire()
        self.load_bart()
        self.load_cct_cold()
        self.load_cct_hot()
        self.load_igt()
        return self

    def load_users(self):
        path = f"{self.data_dir}/users.xlsx"
        self.users = pd.read_excel(path)

    def load_questionnaire(self):
        path = f"{self.data_dir}/questionnaire_responses.csv"
        self.q_responses = pd.read_csv(path)

    def load_bart(self):
        self.bart = pd.read_csv(f"{self.data_dir}/bart_results.csv")

    def load_cct_cold(self):
        self.cct_cold = pd.read_csv(f"{self.data_dir}/cct_cold_results.csv")

    def load_cct_hot(self):
        self.cct_hot = pd.read_csv(f"{self.data_dir}/cct_hot_results.csv")

    def load_igt(self):
        self.igt = pd.read_csv(f"{self.data_dir}/igt_results.csv")