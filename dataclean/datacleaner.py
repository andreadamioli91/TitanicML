import pandas as pd


class DataCleaner:

    def fill_na(self, df):
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
        for column in numeric_columns:
            df[column].fillna(df[column].mean(), inplace=True)
        return df

    def drop_useless(self, df):
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
        correlations = df[numeric_columns].corr()['Survived'].drop('Survived')
        columns_to_drop = correlations[abs(correlations) < 0.05].index
        df.drop(['Embarked', 'Name'], axis=1, inplace=True)
        df.drop(columns_to_drop, axis=1, inplace=True)
        return df

    def convert_dummies(self, df):
        df = pd.get_dummies(df, columns=['Sex'], drop_first=True)
        pass