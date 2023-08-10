import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

class LogisticRegressionTrainer(BaseEstimator, TransformerMixin):
    def __init__(self, penalty='l2', C=1.0):
        self.penalty = penalty
        self.C = C
        self.model = LogisticRegression(penalty=self.penalty, C=self.C)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        predictions = self.model.predict(X)
        return predictions

    def cross_val_score(self, X, y, cv=3, scoring="accuracy"):
        cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring=scoring)
        return cv_scores