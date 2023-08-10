import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

class RandomForestCVTrainer(BaseEstimator, TransformerMixin):
    def __init__(self, n_estimators=100, random_state=None):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = RandomForestClassifier(n_estimators=self.n_estimators, random_state=self.random_state)

    def fit(self, X, y=None):
        # Train the Random Forest model using cross-validation
        self.model.fit(X, y)
        return self

    def transform(self, X):
        # This method is not used in this case, but required for TransformerMixin
        return X

    def predict(self, X):
        # Make predictions using the trained Random Forest model
        return self.model.predict(X)

    def predict_proba(self, X):
        # Return class probabilities using the trained Random Forest model
        return self.model.predict_proba(X)

    def cross_val_score(self, X, y, cv=5, scoring=None):
        # Perform cross-validation and return the cross-validated scores
        scores = cross_val_score(self.model, X, y, cv=cv, scoring=scoring)
        return scores