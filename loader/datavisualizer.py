import pandas as pd


class DataVisualizer:

    def print_dataframe(self, df, max_rows, max_columns):
        pd.set_option('display.max_rows', max_rows)  # Set to None to display all rows
        pd.set_option('display.max_columns', max_columns) # Set to None to display all columns
        print(df)
