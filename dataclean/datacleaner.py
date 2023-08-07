import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


class DataCleaner:

    def fill_na(self, df):
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
        imputer = IterativeImputer(max_iter=10, random_state=0)
        # Fit the imputer on the DataFrame and transform it to fill the missing values
        filled_data = imputer.fit_transform(df)
        # Convert the filled data back to a DataFrame with the original columns
        filled_df = pd.DataFrame(filled_data, columns=df.columns)
        return filled_df

    def drop_useless(self, df):
        df.drop(['Embarked', 'Name'], axis=1, inplace=True)
        return df

    def drop_low_correlated(self, df):
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
        correlations = df[numeric_columns].corr()['Survived'].drop('Survived')
        columns_to_drop = correlations[abs(correlations) < 0.05].index
        df.drop(columns_to_drop, axis=1, inplace=True)
        return df

    def convert_dummies(self, df):
        df = pd.get_dummies(df, columns=['Sex'], drop_first=True)
        df = pd.get_dummies(df, columns=['Pclass'], drop_first=True)
        df = pd.get_dummies(df, columns=['Cabin_Type'], drop_first=True)
        return df

    def transform_complex_columns(self, df):
        df = self.manage_cabin(df)
        df = self.manage_ticket(df)
        return df

    def manage_cabin(self, df):
        # Extract the first letter from the "Cabin" column, that should be a type
        df['Cabin_Type'] = df['Cabin'].apply(lambda x: x[0] if isinstance(x, str) else None)
        # Check if the passenger has more than on cabins
        df['Cabin_Is_Multiple'] = df['Cabin'].apply(lambda x: len(x) > 4 if isinstance(x, str) else None)
        df.drop(['Cabin'], axis=1, inplace=True)
        return df

    def manage_ticket(self, df):
        # Check if the "Ticket" value contains only numbers
        df['Is_Numerical_Ticket'] = df['Ticket'].apply(lambda x: x.isdigit() if isinstance(x, str) else None)
        # Check if the last cipher in the "Ticket" value is even
        df['Last_Cypher_Is_Even'] = df['Ticket'].apply(
            lambda x: int(str(x)[-1]) % 2 == 0 if isinstance(x, str) and x.isdigit() else None)
        df.drop(['Ticket'], axis=1, inplace=True)
        return df

    def manage_formats(self, df):
        # Fill NaN values with a default value (e.g., -1)
        df.fillna(-1, inplace=True)

        df = df.astype(int)
        return df
