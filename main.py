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
    df = loader.load_data(TRAIN_DATASET)
    df = data_cleaner.drop_useless(df)
    df = data_cleaner.transform_complex_columns(df)
    df = data_cleaner.convert_dummies(df)
    df = data_cleaner.fill_na(df)
    df = data_cleaner.manage_formats(df)
    data_visualizer.show_correlation_heatmap(df, "Survived")
    # data_visualizer.print_dataframe(dataset, 5, None)
    print("End of the script")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
