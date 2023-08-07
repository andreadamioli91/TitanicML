import pandas as pd

from loader.csvloader import CsvLoader
from loader.datavisualizer import DataVisualizer
from sklearn.pipeline import Pipeline
from transformer.fillna import FillNa
from transformer.uselessdrop import UselessDrop
from transformer.transformComplex import TransformComplex
from transformer.dummies import Dummies
from transformer.typeconverter import TypeConverter
from transformer.lowcorrelateddrop import DropLowCorrelated
from sklearn.preprocessing import StandardScaler, FunctionTransformer

TRAIN_DATASET = "data/train.csv"
TEST_DATASET = "data/test.csv"
MIN_CORRELATION = 0.05

loader = CsvLoader()
data_visualizer = DataVisualizer()

# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    transformer_pipeline = Pipeline([
        ("fillNa", FillNa()),
        ("uselessDrop", UselessDrop()),
        ("transformComplex", TransformComplex()),
        ("dummies", Dummies()),
        ("typeConvert", TypeConverter()),
        ("dropLowCorrelated", DropLowCorrelated(MIN_CORRELATION)),
    ])
    X_initial = loader.load_data(TRAIN_DATASET)
    X_transformed = transformer_pipeline.fit_transform(X_initial)

    scaler_pipeline = Pipeline([
        ('standardScaler', StandardScaler()),
        ('toDataFrame', FunctionTransformer(lambda x: pd.DataFrame(x, columns=X_transformed.columns)))
    ])
    X_scaled = scaler_pipeline.fit_transform(X_transformed)

    data_visualizer.show_correlation_heatmap(X_transformed, "Survived")
    data_visualizer.print_dataframe(X_transformed, 5, None)
print("End of the script")
