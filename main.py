import pandas as pd

from loader.csvloader import CsvLoader
from loader.datavisualizer import DataVisualizer
from sklearn.pipeline import Pipeline

from model.randomforest.randomforest import RandomForestCVTrainer
from model.randomforest.randomforestexecution import RandomForestExecution
from model.xgboost.xgboostexecution import XGBoostExecution
from transformer.fillna import FillNa
from transformer.uselessdrop import UselessDrop
from transformer.transformComplex import TransformComplex
from transformer.dummies import Dummies
from transformer.typeconverter import TypeConverter
from transformer.lowcorrelateddrop import FindLowCorrelated
from sklearn.preprocessing import StandardScaler, FunctionTransformer

from utils.Utils import Utils

TRAIN_DATASET = "data/train.csv"
TEST_DATASET = "data/test.csv"
MIN_CORRELATION = 0.05

loader = CsvLoader()
data_visualizer = DataVisualizer()
low_correlated = FindLowCorrelated()
utils = Utils()

random_forest = RandomForestExecution()
xgb = XGBoostExecution()


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    train_initial = loader.load_data(TRAIN_DATASET)
    test_initial = loader.load_data(TEST_DATASET)
    low_correlated_columns = low_correlated.get_low_correlated_cols(train_initial, MIN_CORRELATION)
    know_useless_columns = ["Embarked", "Name"]

    transformer_pipeline = Pipeline([
        ("fillNa", FillNa()),
        ("dropUseless", UselessDrop(know_useless_columns)),
        ("dropUncorrelated", UselessDrop(low_correlated_columns)),
        ("transformComplex", TransformComplex()),
        ("dummies", Dummies()),
        ("typeConvert", TypeConverter()),
    ])

    train_transformed = transformer_pipeline.fit_transform(train_initial)
    test_transformed = transformer_pipeline.fit_transform(test_initial)

    train_y = train_transformed["Survived"].astype(int)  # y contains only the "Survived" column
    train_transformed = train_transformed.drop(columns=["Survived"])

    utils.parify_test_columns(train_transformed, test_transformed)

    scaler_pipeline = Pipeline([
        ("standardScaler", StandardScaler()),
        ("toDataFrame", FunctionTransformer(lambda x: pd.DataFrame(x, columns=train_transformed.columns)))
    ])
    train_scaled = scaler_pipeline.fit_transform(train_transformed)
    test_scaled = scaler_pipeline.fit_transform(test_transformed)

    # data_visualizer.show_correlation_heatmap(data_transformed, "Survived")
    # data_visualizer.print_dataframe(data_transformed, 5, None)

    rf_predictions = random_forest.fit_and_predict(train_scaled, train_y, test_scaled)
    xgb_predictions = xgb.fit_and_predict(train_scaled, train_y, test_scaled)

    rf_predictions_solution = pd.DataFrame({"PassengerId": test_initial["PassengerId"], "Survived": rf_predictions})
    xgb_predictions_solution = pd.DataFrame({"PassengerId": test_initial["PassengerId"], "Survived": xgb_predictions})

    # data_visualizer.print_dataframe(predictions_df, 5, None)

    rf_predictions_solution.to_csv('data/rf_solution.csv', index=False)
    xgb_predictions_solution.to_csv('data/xg_solution.csv', index=False)

print("End of the script")
