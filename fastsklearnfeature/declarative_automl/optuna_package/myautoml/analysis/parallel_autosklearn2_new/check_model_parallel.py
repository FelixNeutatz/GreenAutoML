import pickle
import argparse
import openml
from sklearn.metrics import balanced_accuracy_score
from autosklearn.metrics import balanced_accuracy
from autosklearn.experimental.askl2 import AutoSklearn2Classifier
import getpass
import time
import os
import shutil
import traceback
import autosklearn.classification
import sklearn
from multiprocessing import Process, set_start_method, Manager

from anytree import RenderTree
import numpy as np
from codecarbon import OfflineEmissionsTracker

class ConstraintRun(object):
    def __init__(self, space, best_trial, test_score, more=None, tracker=None, tracker_inference=None, len_pred=None):
        self.space = space
        self.best_trial = best_trial
        self.test_score = test_score
        self.more = more
        self.tracker = tracker
        self.tracker_inference = tracker_inference
        self.len_pred = len_pred

    def print_space(self):
        for pre, _, node in RenderTree(self.space.parameter_tree):
            if node.status == True:
                print("%s%s" % (pre, node.name))

    def get_best_config(self):
        return self.best_trial.params

class ConstraintEvaluation(object):
    def __init__(self, dataset=None, constraint=None, system_def=None):
        self.runs = []
        self.dataset = dataset
        self.constraint = constraint
        self.system_def = system_def

    def append(self, constraint_run: ConstraintRun):
        self.runs.append(constraint_run)

    def get_best_run(self):
        max_score = -np.inf
        max_run = None
        for i in range(len(self.runs)):
            if max_score < self.runs[i].test_score:
                max_score = self.runs[i].test_score
                max_run = self.runs[i]
        return max_run

    def get_best_config(self):
        best_run = self.get_best_run()
        return best_run.get_best_config()

    def get_worst_run(self):
        min_score = np.inf
        min_run = None
        for i in range(len(self.runs)):
            if min_score > self.runs[i].test_score:
                min_score = self.runs[i].test_score
                min_run = self.runs[i]
        return min_run

def get_data(data_id, randomstate=42, task_id=None):
    task = None
    if type(task_id) != type(None):
        task = openml.tasks.get_task(task_id)
        data_id = task.get_dataset().dataset_id

    dataset = openml.datasets.get_dataset(dataset_id=data_id)

    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="array",
        target=dataset.default_target_attribute
    )

    X_train = None
    X_test = None
    y_train = None
    y_test = None
    if type(task_id) != type(None):
        train_indices, test_indices = task.get_train_test_split_indices()
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_test = X[test_indices]
        y_test = y[test_indices]
    else:
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,
                                                                                    y,
                                                                                    random_state=randomstate,
                                                                                    stratify=y,
                                                                                    train_size=0.66)

    return X_train, X_test, y_train, y_test, categorical_indicator, attribute_names


openml.config.apikey = '4384bd56dad8c3d2c0f6630c52ef5567'
openml.config.cache_directory = '/home/neutatz/phd2/cache_openml'

# Initiate the parser
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-d", help="OpenML datatset ID", type=int)
parser.add_argument("--outputname", "-o", help="Name of the output file")
args = parser.parse_args()
print(args.dataset)


print(args)

#args.dataset = 31

memory_budget = 500.0
privacy = None

def evaluatePipeline(return_dict):
    categorical_indicator_hold = return_dict['categorical_indicator_hold']
    search_time_frozen = return_dict['search_time_frozen']
    repeat = return_dict['repeat']
    X_train_hold = return_dict['X_train_hold']
    y_train_hold = return_dict['y_train_hold']
    X_test_hold = return_dict['X_test_hold']
    y_test_hold = return_dict['y_test_hold']

    tmp_path = "/home/" + getpass.getuser() + "/data/auto_tmp/autosklearn" + str(time.time()) + '_' + str(
        np.random.randint(1000)) + 'folder'

    try:
        automl = None
        with OfflineEmissionsTracker(save_to_file=False, country_iso_code="CAN", project_name="train"+str(test_holdout_dataset_id)+ str(time.time())) as tracker:

            feat_type = []
            for c_i in range(len(categorical_indicator_hold)):
                if categorical_indicator_hold[c_i]:
                    feat_type.append('Categorical')
                else:
                    feat_type.append('Numerical')

            X_train_sample = X_train_hold
            y_train_sample = y_train_hold


            automl = AutoSklearn2Classifier(
                time_left_for_this_task=search_time_frozen,
                delete_tmp_folder_after_terminate=True,
                metric=balanced_accuracy,
                seed=repeat,
                memory_limit=1024 * 250,
                tmp_folder=tmp_path
            )
            automl.fit(X_train_sample.copy(), y_train_sample.copy(), feat_type=feat_type, metric=balanced_accuracy)


            '''
            automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=search_time_frozen,
                                                                      delete_tmp_folder_after_terminate=True,
                                                                      metric=balanced_accuracy,
                                                                      seed=repeat,
                                                                      memory_limit=1024 * 250,
                                                                      tmp_folder=tmp_path, n_jobs=1)
            automl.fit(X_train_sample.copy(), y_train_sample.copy(), feat_type=feat_type)
            '''

            # automl.refit(X_train_sample.copy(), y_train_sample.copy())

        with OfflineEmissionsTracker(save_to_file=False, country_iso_code="CAN", project_name="inference"+str(test_holdout_dataset_id)+ str(time.time())) as tracker_inference:
            y_hat = automl.predict(X_test_hold)
        result = balanced_accuracy_score(y_test_hold, y_hat)
        return_dict['result'] = result
        return_dict['tracker'] = tracker.final_emissions_data.values
        return_dict['tracker_inference'] = tracker_inference.final_emissions_data.values
        return_dict['len_pred'] = len(X_test_hold)

    except Exception as e:
        traceback.print_exc()
        print(e)
        return_dict['result'] = 0
        return_dict['tracker'] = None
        return_dict['tracker_inference'] = None
        return_dict['len_pred'] = None
    finally:
        if os.path.exists(tmp_path) and os.path.isdir(tmp_path):
            # shutil.rmtree(tmp_path)
            os.system('rm -fr "%s"' % tmp_path)


if __name__ == "__main__":

    for test_holdout_dataset_id in [args.dataset]:

        X_train_hold, X_test_hold, y_train_hold, y_test_hold, categorical_indicator_hold, attribute_names_hold = get_data('data', randomstate=42, task_id=test_holdout_dataset_id)

        dynamic_approach = []

        new_constraint_evaluation_dynamic_all = []

        #for minutes_to_search in [10, 30, 60, 5 * 60]:
        #for minutes_to_search in [30, 60, 5 * 60]:
        for minutes_to_search in [30]:
            # for minutes_to_search in [5 * 60]:

            current_dynamic = []
            search_time_frozen = minutes_to_search  # * 60
            new_constraint_evaluation_dynamic = ConstraintEvaluation(dataset=test_holdout_dataset_id,
                                                                     constraint={'search_time': minutes_to_search},
                                                                     system_def='dynamic')

            for repeat in range(10):

                result = 0.0

                manager = Manager()
                return_dict = manager.dict()
                return_dict['categorical_indicator_hold'] = categorical_indicator_hold
                return_dict['search_time_frozen'] = search_time_frozen
                return_dict['repeat'] = repeat
                return_dict['X_train_hold'] = X_train_hold
                return_dict['y_train_hold'] = y_train_hold
                return_dict['X_test_hold'] = X_test_hold
                return_dict['y_test_hold'] = y_test_hold
                my_process = Process(target=evaluatePipeline, name='start', args=(return_dict,))
                my_process.start()
                #my_process.join(int(minutes_to_search*3))
                my_process.join()

                # If thread is active
                while my_process.is_alive():
                    # Terminate foo
                    my_process.terminate()
                    my_process.join()

                new_constraint_evaluation_dynamic.append(
                    ConstraintRun('test', 'test', return_dict['result'], more='test', tracker=return_dict['tracker'],
                                  tracker_inference=return_dict['tracker_inference'],
                                  len_pred=return_dict['len_pred']))


                current_dynamic.append(return_dict['result'])
                print('dynamic: ' + str(current_dynamic))
            dynamic_approach.append(current_dynamic)
            new_constraint_evaluation_dynamic_all.append(new_constraint_evaluation_dynamic)
            print('dynamic: ' + str(dynamic_approach))

        results_dict_log = {}
        results_dict_log['dynamic'] = new_constraint_evaluation_dynamic_all
        pickle.dump(results_dict_log, open('/home/neutatz/data/automl_runs/log_' + args.outputname + '_' + str(test_holdout_dataset_id) + '.p', 'wb+'))

        results_dict = {}
        results_dict['dynamic'] = dynamic_approach
        pickle.dump(results_dict, open('/home/neutatz/data/automl_runs/' + args.outputname + '_' + str(test_holdout_dataset_id) + '.p', 'wb+'))