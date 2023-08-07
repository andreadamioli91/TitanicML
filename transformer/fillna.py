import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


class FillNa:

    def __init__(self):
        self.imputer = IterativeImputer(max_iter=10)
        self.numeric_columns = []

    def fit(self, df, y=None):
        self.numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
        return self

    def transform(self, df):
        # Select only the numeric columns for imputation
        df_numeric = df[self.numeric_columns]
        # Impute missing values in the numeric columns
        filled_data = self.imputer.fit_transform(df_numeric)
        # Convert the filled data back to a DataFrame with the original columns
        filled_df = pd.DataFrame(filled_data, columns=df_numeric.columns)
        # Merge the imputed numeric columns with the non-numeric columns
        df_filled = pd.concat([df.drop(columns=self.numeric_columns), filled_df], axis=1)
        return df_filled
