import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class BasicNeuralNetwork(BaseEstimator, TransformerMixin):

    def __init__(self, hidden_layers=(128, 64), activation='relu', optimizer='adam', loss='binary_crossentropy',
                 epochs=50, batch_size=32):
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = keras.Sequential([
            layers.Input(16),
            layers.Dense(self.hidden_layers[0], activation=self.activation),
            layers.Dense(self.hidden_layers[1], activation=self.activation),
            layers.Dense(1, activation='sigmoid')
        ])

    def fit(self, X, y):
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        return self

    def predict(self, X):
        predictions = self.model.predict(X)
        binary_predictions = (predictions > 0.5).astype(int)
        return binary_predictions

    def cross_val_score(self, X, y, cv=3, scoring="accuracy"):
        estimator = keras.wrappers.scikit_learn.KerasClassifier(build_fn=self.model_builder, epochs=self.epochs,
                                                                batch_size=self.batch_size, verbose=0)
        cv_scores = cross_val_score(estimator, X, y, cv=cv, scoring=scoring)
        return cv_scores
