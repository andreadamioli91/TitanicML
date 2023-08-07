import pandas as pd

class UselessDrop:

    def fit(self, df, y=None):
        self.cols_to_drop = ['Embarked', 'Name']
        return self

    def transform(self, df):
        df.drop(['Embarked', 'Name'], axis=1, inplace=True)
        return df