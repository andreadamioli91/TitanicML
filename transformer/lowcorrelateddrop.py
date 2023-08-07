class DropLowCorrelated:

    def __init__(self, threshold):
        self.threshold = threshold

    def fit(self, df, y=None):
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
        correlations = df[numeric_columns].corr()['Survived'].drop('Survived')
        self.columns_to_drop = correlations[abs(correlations) < self.threshold].index
        return self

    def transform(self, df):
        df.drop(self.columns_to_drop, axis=1, inplace=True)
        return df
