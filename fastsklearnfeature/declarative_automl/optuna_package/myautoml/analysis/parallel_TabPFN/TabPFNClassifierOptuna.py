from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name
from tabpfn import TabPFNClassifier
import numpy as np
from sklearn.feature_selection import SelectKBest


def variance(X, y=None):
	variances_ = np.var(X, axis=0)
	return variances_

def my_train_test_split(X, y, random_state=42, train_size=100):
    y_vals, y_counts = np.unique(y, return_counts=True)

    class_indices = []
    for y_i in range(len(y_vals)):
        class_indices.append(np.where(y == y_vals[y_i])[0])

    train_ids = []
    test_ids = []
    for row_id in range(train_size):
        for y_i in range(len(y_vals)):
            if row_id < len(class_indices[y_i]):
                if len(train_ids) < train_size:
                    train_ids.append(class_indices[y_i][row_id])
                else:
                    test_ids.append(class_indices[y_i][row_id])
    return X[train_ids], X[test_ids], y[train_ids], y[test_ids]


class TabPFNClassifierOptuna(TabPFNClassifier):

    def init_hyperparameters(self, trial, X, y, name_space=None):
        self.name = id_name('TabPFNClassifier_')

        if type(name_space) != type(None):
            self.name += '_' + name_space

        self.N_ensemble_configurations = 1
        self.device = 'cuda'#'cpu'
        self.kbest = None

    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('TabPFNClassifier_')

    def fit(self, X, y=None, sample_weight=None):
        if X.shape[0] > 1024:
            X, _, y, _ = my_train_test_split(X, y, random_state=42, train_size=1024)
        if X.shape[1] > self.max_num_features:
            self.kbest = SelectKBest(k=self.max_num_features, score_func=variance).fit(X, y)
            X = self.kbest.transform(X)
        return super().fit(X, y)

    def predict_proba(self, X, normalize_with_test=False):
        if X.shape[1] > self.max_num_features:
            X = self.kbest.transform(X)
        return super().predict_proba(X, normalize_with_test=normalize_with_test)

    def custom_iterative_fit(self, X, y=None, sample_weight=None, number_steps=2):
        self.n_estimators = number_steps
        return self.fit(X, y=y)

    def get_max_steps(self):
        return 32