import pandas as pd

class Dummies:

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        df = pd.get_dummies(df, columns=['Sex'], drop_first=True)
        df = pd.get_dummies(df, columns=['Pclass'], drop_first=True)
        df = pd.get_dummies(df, columns=['Cabin_Type'], drop_first=True)
        return df

