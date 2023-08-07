import pandas as pd

class TypeConverter:

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        # Fill NaN values with a default value (e.g., -1)
        df.fillna(-1, inplace=True)
        df = df.astype(int)
        return df