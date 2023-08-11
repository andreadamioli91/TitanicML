from sklearn.preprocessing import LabelEncoder

class ManageLabelEncoding:

    def __init__(self, encodable_columns):
        self.label_encoders = {}
        self.encodable_columns = encodable_columns

    def fit(self, df, y=None):
        for column in self.encodable_columns:
            le = LabelEncoder()
            le.fit(df[column].astype(str))  # Convert to string in case of mixed data types
            self.label_encoders[column] = le
        return self

    def transform(self, df):
        for column in self.encodable_columns:
            le = self.label_encoders[column]
            df[column] = le.transform(df[column].astype(str))
        return df