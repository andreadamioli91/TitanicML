import pandas as pd


class CsvLoader:

    def load_data(self, csv_file):
        try:
            data = pd.read_csv(csv_file)
            print("Data loaded successfully.")
            return data
        except Exception as e:
            print("Error while loading data:", e)
