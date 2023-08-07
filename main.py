from loader.csvloader import CsvLoader
from loader.datavisualizer import DataVisualizer
from dataclean.datacleaner import DataCleaner


TRAIN_DATASET = "data/train.csv"
TEST_DATASET = "data/test.csv"

loader = CsvLoader()
data_cleaner = DataCleaner()
data_visualizer = DataVisualizer()

# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    dataset = loader.load_data(TRAIN_DATASET)
    dataset = data_cleaner.drop_useless(dataset)
    dataset = data_cleaner.transform_complex_columns(dataset)
    dataset = data_cleaner.convert_dummies(dataset)
    dataset = data_cleaner.fill_na(dataset)
    data_visualizer.print_dataframe(dataset, 5, None)
    print("End of the script")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
