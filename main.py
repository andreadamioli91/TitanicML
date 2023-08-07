from loader.csvloader import CsvLoader
from dataclean.datacleaner import  DataCleaner

TRAIN_DATASET = "data/train.csv"
TEST_DATASET = "data/test.csv"

loader = CsvLoader()
datacleaner = DataCleaner()

# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    train_dataset = loader.load_data(TRAIN_DATASET)
    train_dataset = datacleaner.drop_useless(train_dataset)
    train_dataset = datacleaner.convert_dummies(train_dataset)
    train_dataset = datacleaner.fill_na(train_dataset)
    print(train_dataset.head(10))
    print("End of the script")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
