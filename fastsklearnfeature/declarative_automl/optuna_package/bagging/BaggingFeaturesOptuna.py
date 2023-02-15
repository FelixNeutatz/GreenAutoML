from sklearn.base import BaseEstimator, TransformerMixin
from fastsklearnfeature.declarative_automl.optuna_package.optuna_utils import id_name
import numpy as np
from sklearn.model_selection import StratifiedKFold
import copy
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.TabPFNClassifierOptuna import TabPFNClassifierOptuna
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.LinearSVCOptuna import LinearSVCOptuna
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.PassiveAggressiveOptuna import PassiveAggressiveOptuna
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.SGDClassifierOptuna import SGDClassifierOptuna
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.SVCOptuna import SVCOptuna

available_models = [AdaBoostClassifier(),
                            BernoulliNB(),
                            DecisionTreeClassifier(),
                            BernoulliNB(),
                            DecisionTreeClassifier(),
                            ExtraTreesClassifier(),
                            GaussianNB(),
                            HistGradientBoostingClassifier(),
                            KNeighborsClassifier(),
                            LinearDiscriminantAnalysis(),
                            LinearSVCOptuna(),
                            MLPClassifier(),
                            MultinomialNB(),
                            PassiveAggressiveOptuna(),
                            QuadraticDiscriminantAnalysis(),
                            RandomForestClassifier(),
                            SGDClassifierOptuna(),
                            SVCOptuna(),
                            TabPFNClassifierOptuna(device='cpu', N_ensemble_configurations=32)
                            ]

class BaggingFeaturesOptuna(TransformerMixin, BaseEstimator):

    def init_hyperparameters(self, trial, X, y):
        self.name = id_name('BaggingFeatures_')
        self.models = []
        self.test_indices = []
        self.final_ensembles = []
        self.number_splits = 2

        for am in available_models:
            if trial.suggest_categorical(self.name + am.__class__.__name__, [True, False]):
                self.models.append(am)

    def generate_hyperparameters(self, space_gen, depending_node=None):
        self.name = id_name('BaggingFeatures_')

        for am in available_models:
            space_gen.generate_cat(self.name + am.__class__.__name__, [True, False], True, depending_node=depending_node)

    def fit(self, X, y=None):
        for m in range(len(self.models)):
            try:
                trained_models = []
                skf = StratifiedKFold(n_splits=self.number_splits, shuffle=True, random_state=42)
                for train_index, test_index in skf.split(X, y):
                    if m == 0:
                        self.test_indices.append(test_index)
                    current_model = copy.deepcopy(self.models[m])
                    current_model.fit(X[train_index], y[train_index])
                    trained_models.append(current_model)

                voting_ensemble = VotingClassifier(estimators=None, voting='soft')
                voting_ensemble.estimators_ = trained_models
                voting_ensemble.le_ = LabelEncoder().fit(y)
                self.final_ensembles.append(voting_ensemble)
            except:
                pass
        return self

    def fit_transform(self, X, y=None, **fit_params):
        print('fit transform')
        self.fit(X, y=y, **fit_params)

        number_cls = len(self.final_ensembles[0].le_.classes_)
        predictions = np.zeros((len(X), number_cls * len(self.final_ensembles)))
        for model in range(len(self.final_ensembles)):
            for fold in range(len(self.test_indices)):
                predictions[self.test_indices[fold], model*number_cls:(model+1)*number_cls] = self.final_ensembles[model].estimators_[fold].predict_proba(X[self.test_indices[fold]])
        return np.hstack((X, predictions))

    def transform(self, X):
        print('transform')
        number_cls = len(self.final_ensembles[0].le_.classes_)
        predictions = np.zeros((len(X), number_cls * len(self.final_ensembles)))
        for model in range(len(self.final_ensembles)):
            predictions[:, model*number_cls:(model+1)*number_cls] = self.final_ensembles[model].predict_proba(X)
        return np.hstack((X, predictions))