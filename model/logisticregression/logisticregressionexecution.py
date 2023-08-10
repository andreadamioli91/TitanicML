from sklearn.pipeline import Pipeline

from model.logisticregression.logisticregression import LogisticRegressionTrainer


class LogisticRegressionExecution:

    def __init__(self):
        self.model = LogisticRegressionTrainer(penalty='l2', C=1.0)

    def fit_and_predict(self, train_X, train_Y, text_X):
        model_pipeline = Pipeline([
            ("logisticRegression", self.model)
        ])
        # Perform cross-validation and get cross-validated scores
        cv_scores = model_pipeline.named_steps["logisticRegression"].cross_val_score(train_X, train_Y, cv=5,
                                                                               scoring="accuracy")
        print("Mean accuracy:", cv_scores.mean())
        model_pipeline.fit(train_X, train_Y)
        predictions = model_pipeline.predict(text_X)
        return predictions