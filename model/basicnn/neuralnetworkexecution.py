from sklearn.pipeline import Pipeline

from model.basicnn.basicneuralnetwork import BasicNeuralNetwork


class NeuralNetworkExecution:

    def __init__(self):
        self.model = BasicNeuralNetwork(hidden_layers=(128, 64), activation='relu', optimizer='adam', loss='binary_crossentropy', epochs=50, batch_size=32)

    def fit_and_predict(self, train_X, train_Y, text_X):
        model_pipeline = Pipeline([
            ("basicNN", self.model)
        ])
        # Perform cross-validation and get cross-validated scores
        # cv_scores = model_pipeline.named_steps["basicNN"].cross_val_score(train_X, train_Y, cv=5, scoring="accuracy")
        # print("Mean accuracy:", cv_scores.mean())
        model_pipeline.fit(train_X, train_Y)
        predictions = model_pipeline.predict(text_X)
        return predictions