import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin


class XGBoostTrainer(BaseEstimator, TransformerMixin):

    def __init__(self, params=None, num_boost_round=100, threshold=0.5):
        self.params = params if params is not None else {}
        self.num_boost_round = num_boost_round
        self.threshold = threshold
        self.model = None

    def fit(self, X, y):
        dtrain = xgb.DMatrix(data=X, label=y)
        self.model = xgb.train(self.params, dtrain, self.num_boost_round)
        return self

    def predict(self, X):
        dtest = xgb.DMatrix(data=X)
        predictions = self.model.predict(dtest)
        binary_predictions = (predictions > self.threshold).astype(int)
        return binary_predictions

    def cross_val_score(self, X, y, cv=3, scoring=None):
        dtrain = xgb.DMatrix(data=X, label=y)
        cv_scores = xgb.cv(self.params, dtrain, num_boost_round=self.num_boost_round,
                           nfold=cv, metrics=scoring, stratified=True, shuffle=True)
        return cv_scores
