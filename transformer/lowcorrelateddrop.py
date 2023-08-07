class FindLowCorrelated:

    def get_low_correlated_cols(self, df, threshold=0.05):
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
        correlations = df[numeric_columns].corr()['Survived'].drop('Survived')
        columns_to_drop = correlations[abs(correlations) < threshold].index
        return columns_to_drop
