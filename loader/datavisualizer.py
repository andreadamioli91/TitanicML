import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class DataVisualizer:

    def print_dataframe(self, df, max_rows, max_columns):
        pd.set_option('display.max_rows', max_rows)  # Set to None to display all rows
        pd.set_option('display.max_columns', max_columns) # Set to None to display all columns
        print(df)

    def show_correlation_heatmap(self, df, target_column):
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in the DataFrame.")

        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
        data_numeric = df[numeric_columns]

        # Calculate the correlation matrix
        correlation_matrix = data_numeric.corr()

        # Create the heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title(f"Correlation Heatmap with '{target_column}'")
        plt.show()

