from sklearn.pipeline import Pipeline

from model.xgboost.xgboost import XGBoostTrainer


class XGBoostExecution:

    def __init__(self):
        self.model = XGBoostTrainer(params={"objective": "binary:logistic"}, num_boost_round=100, threshold=0.5)

    def fit_and_predict(self, train_X, train_Y, text_X):
        model_pipeline = Pipeline([
            ("xgBoost", self.model)
        ])

        cv_scores = model_pipeline.named_steps["xgBoost"].cross_val_score(train_X, train_Y, cv=5,
                                                                          scoring="logloss")
        print("Mean accuracy:", cv_scores.mean())
        model_pipeline.fit(train_X, train_Y)
        predictions = model_pipeline.predict(text_X)
        return predictions
