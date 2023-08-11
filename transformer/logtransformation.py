import numpy as np


class LogTransformation:

    def __init__(self):
        self.suitable_columns = []

    def fit(self, df, y=None):
        self.suitable_columns = ["Fare"]
        return self

    def transform(self, df):
        for c in self.suitable_columns:
            df[c] = np.log1p(df[c])
        return df