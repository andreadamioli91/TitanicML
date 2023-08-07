from loader.csvloader import CsvLoader
from loader.datavisualizer import DataVisualizer
from sklearn.pipeline import Pipeline
from transformer.fillna import FillNa
from transformer.uselessdrop import UselessDrop
from transformer.transformComplex import TransformComplex
from transformer.dummies import Dummies
from transformer.typeconverter import TypeConverter
from transformer.lowcorrelateddrop import DropLowCorrelated

TRAIN_DATASET = "data/train.csv"
TEST_DATASET = "data/test.csv"

loader = CsvLoader()
data_visualizer = DataVisualizer()

# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    pipeline = Pipeline([
        ("fillNa", FillNa()),
        ("uselessDrop", UselessDrop()),
        ("transformComplex", TransformComplex()),
        ("dummies", Dummies()),
        ("typeConvert", TypeConverter()),
        ("dropLowCorrelated", DropLowCorrelated())
    ])

    df = loader.load_data(TRAIN_DATASET)

    X_transformed = pipeline.fit_transform(df)

    # data_visualizer.show_correlation_heatmap(df, "Survived")
    data_visualizer.print_dataframe(X_transformed, 5, None)
    print("End of the script")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
