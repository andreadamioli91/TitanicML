from sklearn.pipeline import Pipeline

from model.randomforest.randomforest import RandomForestCVTrainer


class RandomForestExecution:

    def __init__(self):
        self.model = RandomForestCVTrainer(n_estimators=100, random_state=42)

    def fit_and_predict(self, train_X, train_Y, text_X):
        model_pipeline = Pipeline([
            ("randomForest", self.model)
        ])
        # Perform cross-validation and get cross-validated scores
        cv_scores = model_pipeline.named_steps["randomForest"].cross_val_score(train_X, train_Y, cv=5,
                                                                               scoring="accuracy")
        print("Mean accuracy:", cv_scores.mean())
        model_pipeline.fit(train_X, train_Y)
        predictions = model_pipeline.predict(text_X)
        return predictions
