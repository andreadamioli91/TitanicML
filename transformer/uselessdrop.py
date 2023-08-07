import pandas as pd

class UselessDrop:

    def __init__(self, cols_to_drop):
        self.cols_to_drop = cols_to_drop

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        df.drop(self.cols_to_drop, axis=1, inplace=True)
        return df