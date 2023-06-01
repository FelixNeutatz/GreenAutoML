import optuna
from imblearn.pipeline import Pipeline
import sklearn.metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
import openml
import numpy as np
from sklearn.compose import ColumnTransformer
from fastsklearnfeature.declarative_automl.optuna_package.data_preprocessing.SimpleImputerOptuna import SimpleImputerOptuna
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.QuadraticDiscriminantAnalysisOptuna import QuadraticDiscriminantAnalysisOptuna
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.KNeighborsClassifierOptuna import KNeighborsClassifierOptuna
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.MLPClassifierOptuna import MLPClassifierOptuna
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.private.PrivateLogisticRegressionOptuna import PrivateLogisticRegressionOptuna
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.private.PrivateGaussianNBOptuna import PrivateGaussianNBOptuna
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.my_system.Space_GenerationTreeBalance import SpaceGenerator
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.LinearDiscriminantAnalysisOptuna import LinearDiscriminantAnalysisOptuna
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.multiclass.OneVsRestClassifierOptuna import OneVsRestClassifierOptuna
from fastsklearnfeature.declarative_automl.optuna_package.data_preprocessing.categorical_encoding.LabelEncoderOptuna import LabelEncoderOptuna
from fastsklearnfeature.declarative_automl.optuna_package.data_preprocessing.categorical_encoding.OneHotEncoderOptuna import OneHotEncoderOptuna
#from fastsklearnfeature.declarative_automl.optuna_package.bagging.BaggingFeaturesOptuna import BaggingFeaturesOptuna
from codecarbon import EmissionsTracker
from sklearn.preprocessing import minmax_scale

import pandas as pd
import time
import resource
import copy
import fastsklearnfeature.declarative_automl.optuna_package.myautoml.define_space as myspace
import pickle
from dataclasses import dataclass
import sys
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.my_system.fairness.metric import true_positive_rate_score
from autosklearn.ensembles.ensemble_selection import EnsembleSelection
from autosklearn.metrics import balanced_accuracy as ba
from autosklearn.constants import MULTICLASS_CLASSIFICATION
import traceback
from multiprocessing import Process, set_start_method, Manager

try:
     set_start_method('fork', force=True)
except RuntimeError:
    pass

from fastsklearnfeature.declarative_automl.optuna_package.myautoml.my_system.ensemble.EnsembleClassifier import EnsembleClassifier
from fastsklearnfeature.declarative_automl.optuna_package.feature_preprocessing.MyIdentity import IdentityTransformation


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

def my_train_test_split_ids(X, y, random_state=42, train_size=100):
    y_vals, y_counts = np.unique(y, return_counts=True)

    class_indices = []
    for y_i in range(len(y_vals)):
        myids = np.where(y == y_vals[y_i])[0]
        #print(myids)

        np.random.seed(seed=random_state)
        np.random.shuffle(myids)

        class_indices.append(myids)

    train_ids = []
    test_ids = []
    for row_id in range(train_size):
        for y_i in range(len(y_vals)):
            if row_id < len(class_indices[y_i]):
                if len(train_ids) < train_size:
                    train_ids.append(class_indices[y_i][row_id])
                else:
                    test_ids.append(class_indices[y_i][row_id])
    return train_ids

@dataclass
class StopWhenOptimumReachedCallback:
    optimum: float

    def __call__(self, study, trial):
        if study.best_value == self.optimum:
            study.stop()

@dataclass
class StopWhenEnergyReachedCallback:
    consumed_energy_limit: float
    tracker: EmissionsTracker

    def __call__(self, study, trial):

        self.tracker._measure_power_and_energy()
        emissions_data = self.tracker._prepare_emissions_data()
        if emissions_data.energy_consumed > self.consumed_energy_limit:
            study.stop()

class TimeException(Exception):
    def __init__(self, message="Time is over!"):
        self.message = message
        super().__init__(self.message)


def constraints_satisfied(p, return_dict, key, training_time, training_time_limit, pipeline_size_limit, inference_time_limit, X, y, group_id=None, fairness_limit=None, inference_energy_limit=None):

    return_dict[key + 'result' + '_fairness'] = -1
    if type(group_id) != type(None):
        fairness = 1 - true_positive_rate_score(pd.DataFrame(y), p.predict(X), sensitive_data=X[:, group_id])
        return_dict[key + 'result' + '_fairness'] = fairness
        if type(fairness_limit) != type(None) and fairness < fairness_limit:
            return_dict[key + 'result'] = -1 * (fairness_limit - fairness)  # return the difference to satisfying the constraint
            return False

    return_dict[key + 'result' + '_training_time'] = training_time
    if type(training_time_limit) != type(None) and training_time > training_time_limit:
        return_dict[key + 'result'] = -1 * (
                    training_time - training_time_limit)  # return the difference to satisfying the constraint
        return False

    dumped_obj = pickle.dumps(p)
    pipeline_size = sys.getsizeof(dumped_obj)
    return_dict[key + 'result' + '_pipeline_size'] = pipeline_size
    if type(pipeline_size_limit) != type(None) and pipeline_size > pipeline_size_limit:
        return_dict[key + 'result'] = -1 * (
                    pipeline_size - pipeline_size_limit)  # return the difference to satisfying the constraint
        return False

    if type(inference_energy_limit) != type(None):
        random_ids = np.random.randint(size=10, low=0, high=X.shape[0])
        tracker_inference = EmissionsTracker(save_to_file=False)
        tracker_inference.start()
        for i in range(10):
            p.predict(X[[random_ids[i]]])
        tracker_inference.stop()
        return_dict[key + 'result' + '_inference_energy'] = tracker_inference.final_emissions_data.values['energy_consumed']
        print('test: ' + str(return_dict[key + 'result' + '_inference_energy']))
        if return_dict[key + 'result' + '_inference_energy'] > inference_energy_limit:
            return_dict[key + 'result'] = -1 * (return_dict[key + 'result' + '_inference_energy'] - inference_energy_limit)  # return the difference to satisfying the constraint
            return False

    if type(inference_time_limit) != type(None):
        inference_times = []
        for i in range(10):
            random_id = np.random.randint(low=0, high=X.shape[0])
            start_inference = time.time()
            p.predict(X[[random_id]])
            inference_times.append(time.time() - start_inference)
        return_dict[key + 'result' + '_inference_time'] = np.mean(inference_times)
        print('inference_time: ' + str(return_dict[key + 'result' + '_inference_time']))
        if return_dict[key + 'result' + '_inference_time'] > inference_time_limit:
            return_dict[key + 'result'] = -1 * (return_dict[key + 'result' + '_inference_time'] - inference_time_limit)  # return the difference to satisfying the constraint
            return False
    return True

def are_monotonic_constraints_satisfied(return_dict, key, training_time_limit, pipeline_size_limit, inference_time_limit, inference_energy_limit):
    if type(training_time_limit) != type(None):
        if key + 'result' + '_training_time' in return_dict and return_dict[key + 'result' + '_training_time'] > training_time_limit:
            return False

    if type(pipeline_size_limit) != type(None):
        if key + 'result' + '_pipeline_size' in return_dict and return_dict[key + 'result' + '_pipeline_size'] > pipeline_size_limit:
            return False

    if type(inference_time_limit) != type(None):
        if key + 'result' + '_inference_time' in return_dict and return_dict[key + 'result' + '_inference_time'] > inference_time_limit:
            return False

    if type(inference_energy_limit) != type(None):
        if key + 'result' + '_inference_energy' in return_dict and return_dict[key + 'result' + '_inference_energy'] > inference_energy_limit:
            return False

    return True


def has_iterative_fit(p):
    if isinstance(p.steps[-1][-1], OneVsRestClassifierOptuna):
        return p.steps[-1][-1].has_iterative_fit()
    else:
        invert_op = getattr(p.steps[-1][-1], "custom_iterative_fit", None)
        return type(invert_op) != type(None)


def ensemble(return_dict):
    model_store = return_dict['model_store']
    val_true_full = return_dict['val_true']

    new_my_keys = list(model_store.keys())
    sorted_keys = np.array(sorted(new_my_keys))
    accuracies_for_keys = np.array([model_store[run_key][1] for run_key in sorted_keys])

    sorted_ids = np.argsort(accuracies_for_keys * -1)
    desc_sorted_keys_full = sorted_keys[sorted_ids]

    sorted_accuracies = accuracies_for_keys[sorted_ids]
    print('accuracies: ' + str(sorted_accuracies))

    #print('accuracies: ' + str(accuracies_for_keys[sorted_ids]))
    #print(desc_sorted_keys)

    for size_ensemble in range(2, len(desc_sorted_keys_full)):

        desc_sorted_keys = desc_sorted_keys_full[:size_ensemble]

        validation_predictions_new = []
        model_list = []

        for k in desc_sorted_keys:
            if float(model_store[k][1]) / sorted_accuracies[0] > 0.50:
                validation_predictions_new.append(model_store[k][3])
                model_list.append(model_store[k][0])


        current_size = len(np.unique(val_true_full)) * 10
        full_size_reached = False
        while not full_size_reached:
            #print('current: ' + str(current_size))
            val_pred = []
            val_true = None
            if current_size < len(val_true_full):
                train_ids = my_train_test_split_ids(np.array(validation_predictions_new), val_true_full, random_state=42,
                                                             train_size=current_size)

                val_true = val_true_full[train_ids]
                for k in desc_sorted_keys:
                    val_pred.append(model_store[k][3][train_ids])

            else:
                val_pred = validation_predictions_new
                val_true = val_true_full
                full_size_reached = True

            ensemble_sel = EnsembleSelection(ensemble_size=len(validation_predictions_new),
                                             task_type=MULTICLASS_CLASSIFICATION,
                                             random_state=0,
                                             metric=ba)
            ensemble_sel.fit(val_pred, val_true, identifiers=None)

            #print(ensemble_sel.weights_)

            current_size *= 3
            current_size = min(current_size, len(val_true_full))

            return_dict['ensemble_store'] = EnsembleClassifier(models=model_list, ensemble_selection=ensemble_sel)

def evaluatePipeline(key, return_dict):
    try:
    #if True:
        p = return_dict['p']
        number_of_cvs = return_dict['number_of_cvs']
        cv = return_dict['cv']
        training_sampling_factor = return_dict['training_sampling_factor']
        scorer = return_dict['scorer']
        X = return_dict['X']
        y = return_dict['y']
        main_memory_budget_gb = return_dict['main_memory_budget_gb']
        hold_out_fraction = return_dict['hold_out_fraction']

        training_time_limit = return_dict['training_time_limit']
        inference_time_limit = return_dict['inference_time_limit']
        inference_energy_limit = return_dict['inference_energy_limit']
        pipeline_size_limit = return_dict['pipeline_size_limit']

        #consumed_energy_limit = return_dict['consumed_energy_limit']
        #tracker = return_dict['tracker']

        adversarial_robustness_constraint = return_dict['adversarial_robustness_constraint']
        fairness_limit = return_dict['fairness_limit']
        group_id = return_dict['fairness_group_id']
        use_incremental_data = return_dict['use_incremental_data']
        shuffle_validation = return_dict['shuffle_validation']
        train_best_with_full_data = return_dict['train_best_with_full_data']

        max_ensemble_models = return_dict['max_ensemble_models']
        model_store = return_dict['model_store']

        dummy_result = return_dict['dummy_result']

        caruana_ensemble = return_dict['caruana_ensemble']

        trial = None
        if 'trial' in return_dict:
            trial = return_dict['trial']

        size = int(main_memory_budget_gb * 1024.0 * 1024.0 * 1024.0)
        resource.setrlimit(resource.RLIMIT_AS, (size, resource.RLIM_INFINITY))

        my_random_state = 42
        if shuffle_validation:
            my_random_state = int(time.time())

        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=my_random_state, stratify=y,
                                                                                    test_size=hold_out_fraction)

        if training_sampling_factor < 1.0:
            X_train, _, y_train, _ = sklearn.model_selection.train_test_split(X_train, y_train,
                                                                              random_state=42,
                                                                              stratify=y_train,
                                                                              train_size=training_sampling_factor)

        X_train_big = copy.deepcopy(X_train)
        y_train_big = copy.deepcopy(y_train)

        current_size = len(np.unique(y_train)) * 10
        if not use_incremental_data:
            current_size = len(X_train_big)

        full_size_reached = False
        current_size_iter = 0

        return_dict[key + 'intermediate_results'] = {}
        #print('trial: ' + str(trial))

        p_new = copy.deepcopy(p)

        while not full_size_reached:
            print('current: ' + str(current_size))

            '''
            if type(None) != type(consumed_energy_limit):
                tracker._measure_power_and_energy()
                emissions_data = tracker._prepare_emissions_data()
                if emissions_data.energy_consumed > consumed_energy_limit:
                    return
            '''


            if current_size < len(X_train_big):
                X_train, _, y_train, _ = my_train_test_split(X_train_big, y_train_big, random_state=42, train_size=current_size)
                current_size *= 3
                current_size = min(current_size, len(X_train_big))
            else:
                X_train = X_train_big
                y_train = y_train_big
                full_size_reached = True

            y_class_counts = np.unique(y_train, return_counts=True)[1]
            print('class_counts: ' + str(y_class_counts))
            p = copy.deepcopy(p_new)
            if np.all(y_class_counts == y_class_counts[0]):
                for element_step_i in range(len(p.steps)):
                    if p.steps[element_step_i][0] == 'augmentation':
                        p.steps.pop(element_step_i)
                        break

            #print('new steps: ' + str(p.steps))


            if not has_iterative_fit(p):
                scores = []
                start_training = time.time()
                p.fit(X_train, y_train)
                training_time = time.time() - start_training
                scores.append(scorer(p, X_test, pd.DataFrame(y_test)))

                new_return_dict = {}
                if constraints_satisfied(p,
                                             new_return_dict,
                                             key,
                                             training_time,
                                             training_time_limit,
                                             pipeline_size_limit,
                                             inference_time_limit,
                                             X,
                                             y,
                                             group_id,
                                             fairness_limit,
                                             inference_energy_limit):
                    if not key + 'result' in return_dict or (
                            key + 'result' in return_dict and np.mean(scores) > return_dict[key + 'result']):
                        return_dict[key + 'pipeline'] = p
                        return_dict[key + 'result'] = np.mean(scores)

                        for k_return, val_return in new_return_dict.items():
                            return_dict[k_return] = val_return

                        start_inference_val_time = time.time()
                        return_dict[key + 'val_predictions'] = p.predict_proba(X_test)
                        return_dict[key + 'val_predictions_time'] = time.time() - start_inference_val_time
                        return_dict[key + 'val_true'] = y_test
                        '''
                        if max_ensemble_models > 1 and caruana_ensemble:
                            start_inference_val_time = time.time()
                            return_dict[key + 'val_predictions'] = p.predict_proba(X_test)
                            return_dict[key + 'val_predictions_time'] = time.time() - start_inference_val_time

                            build_ensemble(return_dict, max_ensemble_models, model_store, dummy_result, key,
                                           pipeline_size_limit, training_time_limit, inference_time_limit, group_id,
                                           fairness_limit, X, y, y_test, inference_energy_limit)
                        '''

                if not key + 'pipeline' in return_dict:
                    if not key + 'result' in return_dict or return_dict[key + 'result'] < new_return_dict[key + 'result']:
                        for k_return, val_return in new_return_dict.items():
                            return_dict[k_return] = val_return

                if not are_monotonic_constraints_satisfied(return_dict, key, training_time_limit, pipeline_size_limit, inference_time_limit, inference_energy_limit):
                    return

            else:
                start_training = time.time()
                Xt, yt, fit_params = p._fit(X_train, y_train)
                training_time = time.time() - start_training
                current_steps = 1
                n_steps = int(2 ** current_steps / 2) if current_steps > 1 else 2
                while n_steps < p.steps[-1][-1].get_max_steps():
                    scores = []
                    start_training = time.time()
                    p.steps[-1][-1].custom_iterative_fit(Xt, yt, number_steps=n_steps)
                    training_time += time.time() - start_training
                    scores.append(scorer(p, X_test, pd.DataFrame(y_test)))

                    new_return_dict = {}
                    if constraints_satisfied(p,
                                             new_return_dict,
                                             key,
                                             training_time,
                                             training_time_limit,
                                             pipeline_size_limit,
                                             inference_time_limit,
                                             X,
                                             y,
                                             group_id,
                                             fairness_limit,
                                             inference_energy_limit):
                        if not key + 'result' in return_dict or (key + 'result' in return_dict and np.mean(scores) > return_dict[key + 'result']) :
                            return_dict[key + 'pipeline'] = copy.deepcopy(p)
                            return_dict[key + 'result'] = np.mean(scores)

                            for k_return, val_return in new_return_dict.items():
                                return_dict[k_return] = val_return

                            start_inference_val_time = time.time()
                            return_dict[key + 'val_predictions'] = p.predict_proba(X_test)
                            return_dict[key + 'val_predictions_time'] = time.time() - start_inference_val_time
                            return_dict[key + 'val_true'] = y_test
                            '''
                            if max_ensemble_models > 1 and caruana_ensemble:
                                start_inference_val_time = time.time()
                                return_dict[key + 'val_predictions'] = p.predict_proba(X_test)
                                return_dict[key + 'val_predictions_time'] = time.time() - start_inference_val_time

                                build_ensemble(return_dict, max_ensemble_models, model_store, dummy_result, key,
                                               pipeline_size_limit, training_time_limit, inference_time_limit, group_id,
                                               fairness_limit, X, y, y_test, inference_energy_limit)
                            '''

                    if not key + 'pipeline' in return_dict:
                        if not key + 'result' in return_dict or return_dict[key + 'result'] < new_return_dict[
                            key + 'result']:
                            for k_return, val_return in new_return_dict.items():
                                return_dict[k_return] = val_return

                    if not are_monotonic_constraints_satisfied(return_dict, key, training_time_limit,
                                                               pipeline_size_limit, inference_time_limit, inference_energy_limit):
                        return

                    current_steps += 1
                    n_steps = int(2 ** current_steps / 2) if current_steps > 1 else 2

            if type(trial) != type(None):
                if key + 'result' in return_dict:
                    trial.report(return_dict[key + 'result'], current_size_iter)
                    return_dict[key + 'intermediate_results_' + str(current_size_iter)] = return_dict[key + 'result']
                else:
                    trial.report(-1, current_size_iter)
                    return_dict[key + 'intermediate_results_' + str(current_size_iter)] = -1 * np.inf

                if use_incremental_data and trial.should_prune():
                    print('I prunet it!')
                    return
                else:
                    print('dont prunet it!')

            current_size_iter += 1


        #check if current model is best so far
        if train_best_with_full_data and key + 'result' in return_dict:
            if return_dict[key + 'result'] > return_dict['study_best_value']:
                print('retrain')
                best_model_so_far = copy.deepcopy(return_dict[key + 'pipeline'])
                start_training = time.time()
                best_model_so_far.fit(X, y)
                training_time = time.time() - start_training

                new_return_dict = {}
                if constraints_satisfied(p,
                                         new_return_dict,
                                         key,
                                         training_time,
                                         training_time_limit,
                                         pipeline_size_limit,
                                         inference_time_limit,
                                         X,
                                         y,
                                         group_id,
                                         fairness_limit,
                                         inference_energy_limit):
                    return_dict[key + 'pipeline'] = best_model_so_far

                    for k_return, val_return in new_return_dict.items():
                        return_dict[k_return] = val_return




    except Exception as e:
        print(str(e) + '\n\n')
        traceback.print_exc()





class MyAutoML:
    def __init__(self, cv=5,
                 number_of_cvs=1,
                 hold_out_fraction=None,
                 evaluation_budget=np.inf,
                 time_search_budget=10*60,
                 n_jobs=1,
                 space=None,
                 study=None,
                 main_memory_budget_gb=4,
                 sample_fraction=1.0,
                 differential_privacy_epsilon=None,
                 training_time_limit=None,
                 inference_time_limit=None,
                 inference_energy_limit=None,
                 pipeline_size_limit=None,
                 adversarial_robustness_constraint=None,
                 fairness_limit=None,
                 fairness_group_id=None,
                 max_ensemble_models=50,
                 use_incremental_data=True,
                 shuffle_validation=False,
                 train_best_with_full_data=False,
                 consumed_energy_limit=None,
                 caruana_ensemble=True,
                 time_fraction_ensemble=0.0
                 ):
        self.cv = cv
        self.time_search_budget = time_search_budget
        self.n_jobs = n_jobs
        self.evaluation_budget = evaluation_budget
        self.number_of_cvs = number_of_cvs
        self.hold_out_fraction = hold_out_fraction
        self.use_incremental_data = use_incremental_data
        self.shuffle_validation = shuffle_validation
        self.train_best_with_full_data = train_best_with_full_data

        self.classifier_list = myspace.classifier_list
        self.private_classifier_list = myspace.private_classifier_list
        self.preprocessor_list = myspace.preprocessor_list
        self.scaling_list = myspace.scaling_list
        self.categorical_encoding_list = myspace.categorical_encoding_list
        self.augmentation_list = myspace.augmentation_list

        self.fairness_limit = fairness_limit
        self.fairness_group_id = fairness_group_id

        #generate binary or mapping for each hyperparameter


        self.space = space
        self.study = study
        self.main_memory_budget_gb = main_memory_budget_gb
        self.sample_fraction = sample_fraction
        self.differential_privacy_epsilon = differential_privacy_epsilon

        self.training_time_limit = training_time_limit
        self.inference_time_limit = inference_time_limit
        self.inference_energy_limit = inference_energy_limit
        self.pipeline_size_limit = pipeline_size_limit
        self.consumed_energy_limit = consumed_energy_limit
        self.tracker = None
        if type(None) != type(self.consumed_energy_limit):
            self.tracker = EmissionsTracker(save_to_file=False)


        self.adversarial_robustness_constraint = adversarial_robustness_constraint

        self.random_key = str(time.time()) + '-' + str(np.random.randint(0, 1000))

        self.max_ensemble_models = max_ensemble_models
        self.ensemble_selection = None
        self.caruana_ensemble = caruana_ensemble
        self.ensemble_time = float(time_fraction_ensemble) * self.time_search_budget

        self.time_search_budget = (1.0 - time_fraction_ensemble) * self.time_search_budget

        self.dummy_result = -1


    def get_best_pipeline(self):
        try:
            max_accuracy = -np.inf
            best_pipeline = None
            for k, v in self.model_store.items():
                if v[1] > max_accuracy:
                    max_accuracy = v[1]
                    best_pipeline = v[0]
            return best_pipeline
        except:
            return None


    def predict(self, X):
        try:
            return self.ensemble_store.predict(X)
        except:
            return self.get_best_pipeline().predict(X)


    def fit(self, X_new, y_new, sample_weight=None, categorical_indicator=None, scorer=None):
        self.start_fitting = time.time()

        if type(None) != type(self.consumed_energy_limit):
            self.tracker.start()

        self.model_store = {}
        self.ensemble_store = None

        if self.sample_fraction < 1.0:
            X, _, y, _ = sklearn.model_selection.train_test_split(X_new, y_new, random_state=42, stratify=y_new, train_size=self.sample_fraction)
        else:
            X = X_new
            y = y_new

        def run_dummy(training_sampling_factor):
            dummy_classifier = sklearn.dummy.DummyClassifier()
            my_pipeline = Pipeline([('classifier', dummy_classifier)])

            key = 'My_automl' + self.random_key + 'My_process' + str(time.time()) + "##" + str(
                np.random.randint(0, 1000))

            manager = Manager()
            return_dict = manager.dict()

            return_dict['p'] = copy.deepcopy(my_pipeline)
            return_dict['number_of_cvs'] = self.number_of_cvs
            return_dict['cv'] = self.cv
            return_dict['training_sampling_factor'] = training_sampling_factor
            return_dict['scorer'] = scorer
            return_dict['X'] = X
            return_dict['y'] = y
            return_dict['main_memory_budget_gb'] = self.main_memory_budget_gb
            return_dict['hold_out_fraction'] = self.hold_out_fraction
            return_dict['training_time_limit'] = self.training_time_limit
            return_dict['inference_time_limit'] = self.inference_time_limit
            return_dict['inference_energy_limit'] = self.inference_energy_limit
            return_dict['pipeline_size_limit'] = self.pipeline_size_limit
            #return_dict['consumed_energy_limit'] = self.consumed_energy_limit
            #return_dict['tracker'] = self.tracker
            return_dict['adversarial_robustness_constraint'] = self.adversarial_robustness_constraint
            return_dict['fairness_limit'] = self.fairness_limit
            return_dict['fairness_group_id'] = self.fairness_group_id
            return_dict['use_incremental_data'] = False
            return_dict['shuffle_validation'] = False
            return_dict['train_best_with_full_data'] = False
            return_dict['model_store'] = {}
            return_dict['max_ensemble_models'] = 1
            return_dict['dummy_result'] = 0.0
            return_dict['caruana_ensemble'] = self.caruana_ensemble

            try:
                return_dict['study_best_value'] = self.study.best_value
            except ValueError:
                return_dict['study_best_value'] = -np.inf

            already_used_time = time.time() - self.start_fitting

            remaining_time = np.min([self.evaluation_budget, self.time_search_budget - already_used_time])

            my_process = Process(target=evaluatePipeline, name='start' + key, args=(key, return_dict,))
            my_process.start()
            my_process.join(int(remaining_time))

            # If thread is active
            while my_process.is_alive():
                # Terminate foo
                my_process.terminate()
                my_process.join()

            if key + 'result' in return_dict:
                self.dummy_result = return_dict[key + 'result']
            else:
                self.dummy_result = 0.0

            print('dummy result: ' + str(self.dummy_result))

        def objective1(trial):
            should_be_pruned = False
            start_total = time.time()

            try:
                self.space.trial = trial

                imputer = SimpleImputerOptuna()
                imputer.init_hyperparameters(self.space, X, y)

                scaler = self.space.suggest_categorical('scaler', self.scaling_list)
                scaler.init_hyperparameters(self.space, X, y)

                onehot_transformer = self.space.suggest_categorical('categorical_encoding', self.categorical_encoding_list)
                onehot_transformer.init_hyperparameters(self.space, X, y)

                preprocessor = self.space.suggest_categorical('preprocessor', self.preprocessor_list)
                preprocessor.init_hyperparameters(self.space, X, y)



                if type(self.differential_privacy_epsilon) == type(None):
                    classifier = self.space.suggest_categorical('classifier', self.classifier_list)
                else:
                    classifier = self.space.suggest_categorical('private_classifier', self.private_classifier_list)

                #from fastsklearnfeature.declarative_automl.optuna_package.classifiers.DecisionTreeClassifierOptuna import DecisionTreeClassifierOptuna
                #classifier = DecisionTreeClassifierOptuna()

                classifier.init_hyperparameters(self.space, X, y)

                multi_class_support = self.space.suggest_categorical('multi_class_support', ['default', 'one_vs_rest'])

                class_weighting = False
                custom_weighting = False
                custom_weight = 'balanced'

                if isinstance(classifier, KNeighborsClassifierOptuna) or \
                        isinstance(classifier, QuadraticDiscriminantAnalysisOptuna) or \
                        isinstance(classifier, PrivateLogisticRegressionOptuna) or \
                        isinstance(classifier, PrivateGaussianNBOptuna) or \
                        isinstance(classifier, MLPClassifierOptuna) or \
                        isinstance(classifier, LinearDiscriminantAnalysisOptuna) or \
                        multi_class_support == 'one_vs_rest':
                    pass
                else:
                    class_weighting = self.space.suggest_categorical('class_weighting', [True, False])
                    if class_weighting:
                        custom_weighting = self.space.suggest_categorical('custom_weighting', [True, False])
                        if custom_weighting:
                            unique_counts = np.unique(y)
                            custom_weight = {}
                            for unique_i in range(len(unique_counts)):
                                custom_weight[unique_counts[unique_i]] = self.space.suggest_uniform(
                                    'custom_class_weight' + str(unique_i), 0.0, 1.0, check=False)

                if class_weighting:
                    classifier.set_weight(custom_weight)


                use_training_sampling = self.space.suggest_categorical('use_training_sampling', [True, False])
                training_sampling_factor = 1.0
                if use_training_sampling:
                    training_sampling_factor = self.space.suggest_uniform('training_sampling_factor', 0.0, 1.0)

                if self.dummy_result == -1:
                    run_dummy(training_sampling_factor)

                y_class_counts = np.unique(y, return_counts=True)[1]
                augmentation = IdentityTransformation()
                if not np.all(y_class_counts == y_class_counts[0]):
                    augmentation = self.space.suggest_categorical('augmentation', self.augmentation_list)
                    augmentation.init_hyperparameters(self.space, X, y)

                numeric_transformer = Pipeline([('imputation', imputer), ('scaler', scaler)])

                if isinstance(onehot_transformer, OneHotEncoderOptuna):
                    categorical_transformer = Pipeline([('removeNAN', LabelEncoderOptuna()), ('onehot_transform', onehot_transformer)])
                else:
                    categorical_transformer = Pipeline([('onehot_transform', onehot_transformer)])


                my_transformers = []
                if np.sum(np.invert(categorical_indicator)) > 0:
                    my_transformers.append(('num', numeric_transformer, np.invert(categorical_indicator)))

                if np.sum(categorical_indicator) > 0:
                    my_transformers.append(('cat', categorical_transformer, categorical_indicator))


                data_preprocessor = ColumnTransformer(transformers=my_transformers)

                multiclass_classifier = classifier
                if multi_class_support == 'one_vs_rest':
                    multiclass_classifier = OneVsRestClassifierOptuna(estimator=classifier, n_jobs=1)

                #multiclass_classifier = OneVsRestClassifierOptuna(estimator=classifier, n_jobs=1)

                bagging = IdentityTransformation()
                '''
                if self.space.suggest_categorical('use_bagging', [True, False]):
                    bagging = BaggingFeaturesOptuna()
                    bagging.init_hyperparameters(self.space, X, y)
                '''

                my_pipeline = Pipeline([('data_preprocessing', data_preprocessor), ('preprocessing', preprocessor), ('bagging', bagging), ('augmentation', augmentation),
                              ('classifier', multiclass_classifier)])

                #my_pipeline = Pipeline([('data_preprocessing', data_preprocessor), ('preprocessing', preprocessor),
                #                        ('classifier', multiclass_classifier)])

                key = 'My_automl' + self.random_key + 'My_process' + str(time.time()) + "##" + str(np.random.randint(0,1000))

                manager = Manager()
                return_dict = manager.dict()

                return_dict['p'] = copy.deepcopy(my_pipeline)
                return_dict['number_of_cvs'] = self.number_of_cvs
                return_dict['cv'] = self.cv
                return_dict['training_sampling_factor'] = training_sampling_factor
                return_dict['scorer'] = scorer
                return_dict['X'] = X
                return_dict['y'] = y
                return_dict['main_memory_budget_gb'] = self.main_memory_budget_gb
                return_dict['hold_out_fraction'] = self.hold_out_fraction
                return_dict['training_time_limit'] = self.training_time_limit
                return_dict['inference_time_limit'] = self.inference_time_limit
                return_dict['inference_energy_limit'] = self.inference_energy_limit
                return_dict['pipeline_size_limit'] = self.pipeline_size_limit
                #return_dict['consumed_energy_limit'] = self.consumed_energy_limit
                #return_dict['tracker'] = self.tracker
                return_dict['adversarial_robustness_constraint'] = self.adversarial_robustness_constraint
                return_dict['fairness_limit'] = self.fairness_limit
                return_dict['fairness_group_id'] = self.fairness_group_id
                return_dict['use_incremental_data'] = self.use_incremental_data
                return_dict['shuffle_validation'] = self.shuffle_validation
                return_dict['train_best_with_full_data'] = self.train_best_with_full_data
                return_dict['max_ensemble_models'] = self.max_ensemble_models
                return_dict['model_store'] = self.model_store
                return_dict['dummy_result'] = self.dummy_result
                return_dict['caruana_ensemble'] = self.caruana_ensemble

                return_dict['trial'] = trial

                try:
                    return_dict['study_best_value'] = self.study.best_value
                except ValueError:
                    return_dict['study_best_value'] = -np.inf

                already_used_time = time.time() - self.start_fitting

                if already_used_time + 2 >= self.time_search_budget:  # already over budget
                    time.sleep(2)
                    return -1 * np.inf

                if type(None) != type(self.consumed_energy_limit):
                    self.tracker._measure_power_and_energy()
                    emissions_data = self.tracker._prepare_emissions_data()
                    if emissions_data.energy_consumed > self.consumed_energy_limit:
                        return -1 * np.inf

                remaining_time = np.min([self.evaluation_budget, self.time_search_budget - already_used_time])


                my_process = Process(target=evaluatePipeline, name='start'+key, args=(key, return_dict,))
                my_process.start()
                my_process.join(int(remaining_time))

                # If thread is active
                while my_process.is_alive():
                    # Terminate foo
                    my_process.terminate()
                    my_process.join()

                #del mp_global.mp_store[key]

                result = -1.0 * np.inf
                if key + 'result' in return_dict:
                    result = return_dict[key + 'result']

                training_time_current = -1
                if key + 'result' + '_training_time' in return_dict:
                    trial.set_user_attr('training_time', return_dict[key + 'result' + '_training_time'])
                    training_time_current = return_dict[key + 'result' + '_training_time']
                if key + 'result' + '_pipeline_size' in return_dict:
                    trial.set_user_attr('pipeline_size', return_dict[key + 'result' + '_pipeline_size'])
                if key + 'result' + '_inference_time' in return_dict:
                    trial.set_user_attr('inference_time', return_dict[key + 'result' + '_inference_time'])
                if key + 'result' + '_inference_energy' in return_dict:
                    trial.set_user_attr('inference_energy', return_dict[key + 'result' + '_inference_energy'])


                if key + 'result' + '_fairness' in return_dict:
                    trial.set_user_attr('fairness', return_dict[key + 'result' + '_fairness'])

                trial.set_user_attr('evaluation_time', time.time() - start_total)
                trial.set_user_attr('time_since_start', time.time() - self.start_fitting)

                if key + 'ensemble' in return_dict:
                    self.ensemble_store = return_dict[key + 'ensemble']

                if key + 'val_true' in return_dict:
                    self.val_true = return_dict[key + 'val_true']

                if result > self.dummy_result:
                    if key + 'pipeline' in return_dict:
                        if len(self.model_store) >= self.max_ensemble_models:
                            run_keys = []
                            run_accuracies = []
                            for run_key, run_info in self.model_store.items():
                                run_accuracies.append(run_info[1])
                                run_keys.append(run_key)

                            run_keys = np.array(run_keys)
                            run_accuracies = np.array(run_accuracies)
                            sorted_ids = np.argsort(run_accuracies * -1)

                            to_be_droped = self.max_ensemble_models
                            if result > run_accuracies[sorted_ids][self.max_ensemble_models - 1]:
                                to_be_droped = self.max_ensemble_models - 1
                                val_predictions = None
                                val_predictions_time = None
                                if key + 'val_predictions' in return_dict:
                                    val_predictions = return_dict[key + 'val_predictions']
                                    val_predictions_time = return_dict[key + 'val_predictions_time']
                                self.model_store[key] = (return_dict[key + 'pipeline'], result, training_time_current, val_predictions, val_predictions_time)

                            for drop_key in run_keys[sorted_ids][to_be_droped:]:
                                del self.model_store[drop_key]
                        else:
                            val_predictions = None
                            val_predictions_time = None
                            if key + 'val_predictions' in return_dict:
                                val_predictions = return_dict[key + 'val_predictions']
                                val_predictions_time = return_dict[key + 'val_predictions_time']
                            self.model_store[key] = (return_dict[key + 'pipeline'], result, training_time_current, val_predictions, val_predictions_time)
                        #print('size model store: ' + str(len(self.model_store)))

                if self.use_incremental_data:
                    current_size_iter = 0
                    #print('return: ' + str(return_dict))
                    while key + 'intermediate_results_' + str(current_size_iter) in return_dict:
                        trial.report(return_dict[key + 'intermediate_results_' + str(current_size_iter)], current_size_iter)
                        if trial.should_prune():
                            print('should be pruned')
                            should_be_pruned = True
                            raise optuna.TrialPruned()
                        else:
                            print('should not be pruned')
                        current_size_iter += 1

                return result


            except Exception as e:
                if should_be_pruned:
                    raise optuna.TrialPruned()
                print('Exception: ' + str(e) + '\n\n')
                traceback.print_exc()
                return -1 * np.inf



        if type(self.study) == type(None):
            self.study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner(
                                n_startup_trials=3, n_warmup_steps=0, interval_steps=1
                            ))

        callbacks = []
        if self.max_ensemble_models == 1:
            callbacks.append(StopWhenOptimumReachedCallback(1.0))
        if type(None) != type(self.consumed_energy_limit):
            callbacks.append(StopWhenEnergyReachedCallback(self.consumed_energy_limit, self.tracker))

        self.study.optimize(objective1, timeout=self.time_search_budget,
                            n_jobs=self.n_jobs,
                            catch=(TimeException,),
                            callbacks=callbacks,
                            ) # todo: check for scorer to know what is the optimum


        if len(self.model_store) > 1:
            manager = Manager()
            return_dict = manager.dict()

            return_dict['model_store'] = self.model_store
            return_dict['val_true'] = self.val_true

            my_process = Process(target=ensemble, name='ensemble', args=(return_dict,))
            my_process.start()
            my_process.join(self.ensemble_time)

            # If thread is active
            while my_process.is_alive():
                # Terminate foo
                my_process.terminate()
                my_process.join()

            if 'ensemble_store' in return_dict:
                self.ensemble_store = return_dict['ensemble_store']

        if type(None) != type(self.consumed_energy_limit):
            self.tracker.stop()

        return self.study.best_value




if __name__ == "__main__":
    # auc = make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)
    from sklearn.metrics import balanced_accuracy_score

    auc = make_scorer(balanced_accuracy_score)

    # dataset = openml.datasets.get_dataset(1114)

    #dataset = openml.datasets.get_dataset(1116)
    dataset = openml.datasets.get_dataset(31)  # 51
    #dataset = openml.datasets.get_dataset(40685)
    #dataset = openml.datasets.get_dataset(1596)
    #dataset = openml.datasets.get_dataset(41167)
    #dataset = openml.datasets.get_dataset(41147)
    #dataset = openml.datasets.get_dataset(1596)
    #41167

    #dataset = openml.datasets.get_dataset(41167)

    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format='array',
        target=dataset.default_target_attribute
    )

    #print(X[:, 12])

    #X[:,12] = X[:,12] > np.mean(X[:,12])

    #print(X[:,12])

    print(X.shape)

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=42, stratify=y,
                                                                                train_size=0.6)

    gen = SpaceGenerator()
    space = gen.generate_params()

    # print(space.generate_additional_features_v1(start=True))
    print(len(space.name2node))
    # print(space)
    new_list = list()
    space.generate_additional_features_v2(start=True, sum_list=new_list)
    print(new_list)
    print(len(new_list))

    new_list = list()
    space.generate_additional_features_v2_name(start=True, sum_list=new_list)
    print(new_list)
    print(len(new_list))

    from anytree import RenderTree

    for pre, _, node in RenderTree(space.parameter_tree):
        print("%s%s: %s" % (pre, node.name, node.status))

    '''
    search = MyAutoML(n_jobs=1,
                      time_search_budget=1 * 60,
                      space=space,
                      main_memory_budget_gb=40,
                      hold_out_fraction=0.3,
                      fairness_limit=0.95,
                      fairness_group_id=12)
    '''
    single_perf = []
    ensemble_perf = []

    differences_ens = []

    for _ in range(10):
        search = MyAutoML(n_jobs=1,
                          time_search_budget=2*60,
                          space=space,
                          main_memory_budget_gb=40,
                          hold_out_fraction=0.6,
                          max_ensemble_models=50,
                          use_incremental_data=True,
                          #inference_energy_limit=8.0e-08,
                          #inference_time_limit=0.002,
                          #training_time_limit=0.02
                          #pipeline_size_limit=10000
                          #fairness_limit=0.95,
                          #fairness_group_id=12,
                          shuffle_validation=False,
                          time_fraction_ensemble=0.1

                          )

        begin = time.time()

        best_result = search.fit(X_train, y_train, categorical_indicator=categorical_indicator, scorer=auc)

        print(len(search.model_store))
        #search.ensemble()


        print('time: ' + str(time.time() - begin))
        #print('energy: ' + str(search.tracker.final_emissions_data.energy_consumed))

        #print(search.model_store)

        #from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import show_progress

        #show_progress(search, X_test, y_test, auc)

        # importances = optuna.importance.get_param_importances(search.study)
        # print(importances)


        try:
            test_score = auc(search.get_best_pipeline(), X_test, y_test)
            single_perf.append(test_score)

            tracker_inference = EmissionsTracker(save_to_file=False)
            tracker_inference.start()
            y_hat_test = search.predict(X_test)
            tracker_inference.stop()
            print('inference: ' + str(tracker_inference.final_emissions_data.values['energy_consumed'] / float(len(X_test))))


            #search.ensemble(X_train, y_train)
            y_hat_test = search.predict(X_test)
            print(len(y_test))
            print(len(y_hat_test))

            ensemble_perf.append(balanced_accuracy_score(y_test, y_hat_test))

            print('ensemble result: ' + str(balanced_accuracy_score(y_test, y_hat_test)))
            print(ensemble_perf)
            print("single model: " + str(test_score))
            print(single_perf)

            differences_ens.append(balanced_accuracy_score(y_test, y_hat_test) - test_score)

            print(time.time() - begin)

        except:
            pass

    print('ensemble: ' + str(np.mean(ensemble_perf)) + ' +- ' + str(np.std(ensemble_perf)))
    print('single: ' + str(np.mean(single_perf)) + ' +- ' + str(np.std(single_perf)))
    print('diff: ' + str(np.mean(differences_ens)) + ' +- ' + str(np.std(differences_ens)))