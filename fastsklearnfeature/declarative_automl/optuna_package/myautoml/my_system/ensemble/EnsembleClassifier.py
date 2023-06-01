import numpy as np

class EnsembleClassifier:
    def __init__(self, models, ensemble_selection):
        self.models = []
        for mi in range(len(models)):
            if ensemble_selection.weights_[mi] != 0.0:
                self.models.append(models[mi])
        new_weights = np.zeros((len(self.models),),dtype=np.float64,)
        currenti = 0
        for mi in range(len(models)):
            if ensemble_selection.weights_[mi] != 0.0:
                new_weights[currenti] = ensemble_selection.weights_[mi]
                currenti += 1
        self.ensemble_selection = ensemble_selection
        self.ensemble_selection.weights_ = new_weights
        self.ensemble_selection.num_input_models_ = len(self.models)

    def predict(self, X_test):
        if len(self.models) == 1:
            return self.models[0].predict(X_test)
        else:
            test_predictions = []
            for m in self.models:
                test_predictions.append(m.predict_proba(X_test))
            y_hat_test_temp = self.ensemble_selection.predict(np.array(test_predictions))
            y_hat_test_temp = np.argmax(y_hat_test_temp, axis=1)
            return y_hat_test_temp

    def fit(self, X, y=None):
        for mi in range(len(self.models)):
            self.models[mi].fit(X, y)
        return self