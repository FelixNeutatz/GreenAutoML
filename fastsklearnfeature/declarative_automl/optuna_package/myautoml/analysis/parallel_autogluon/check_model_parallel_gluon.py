from sklearn.metrics import make_scorer
import pickle
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.analysis.parallel.util_classes import ConstraintEvaluation, ConstraintRun
import argparse
import openml
from sklearn.metrics import balanced_accuracy_score
from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
import time
import numpy as np
import getpass
import os
import shutil
from codecarbon import EmissionsTracker
import traceback

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
openml.config.cache_directory = '/home/' + getpass.getuser() + '/phd2/cache_openml'

my_scorer = make_scorer(balanced_accuracy_score)

# Initiate the parser
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-d", help="OpenML datatset ID", type=int)
parser.add_argument("--outputname", "-o", help="Name of the output file")
args = parser.parse_args()
print(args.dataset)


print(args)

#args.dataset = 168794

memory_budget = 500.0
privacy = None

for test_holdout_dataset_id in [args.dataset]:

    X_train_hold, X_test_hold, y_train_hold, y_test_hold, categorical_indicator_hold, attribute_names_hold = get_data('data', randomstate=42, task_id=test_holdout_dataset_id)

    dynamic_approach = []

    new_constraint_evaluation_dynamic_all = []

    for minutes_to_search in [10, 30, 60, 5*60]:
    #for minutes_to_search in [5 * 60]:

        current_dynamic = []
        search_time_frozen = minutes_to_search #* 60
        new_constraint_evaluation_dynamic = ConstraintEvaluation(dataset=test_holdout_dataset_id,
                                                                 constraint={'search_time': minutes_to_search},
                                                                 system_def='dynamic')

        for repeat in range(10):

            tmp_path = "/home/" + getpass.getuser() + "/data/auto_tmp/autosklearn" + str(time.time()) + '_' + str(np.random.randint(1000)) + 'folder'

            try:
                tracker = EmissionsTracker(save_to_file=False)
                tracker.start()
                df = pd.DataFrame(data=X_train_hold)
                label = 'my_target_label1234'
                df[label] = y_train_hold
                my_data_train = TabularDataset(data=df)

                df_test = pd.DataFrame(data=X_test_hold)
                #df_test[label] = y_test_hold
                my_data_test = TabularDataset(data=df_test)

                #presets = ['good_quality_faster_inference_only_refit', 'optimize_for_deployment']
                presets = 'best_quality'

                predictor = TabularPredictor(label=label, eval_metric='balanced_accuracy', path=tmp_path).fit(train_data=my_data_train, time_limit=search_time_frozen, presets=presets, num_cpus='auto', num_gpus=1, ag_args_fit={'num_gpus': 1})#, num_cpus=1, num_gpus='auto'
                tracker.stop()

                tracker_inference = EmissionsTracker(save_to_file=False)
                tracker_inference.start()
                y_hat = predictor.predict(my_data_test)
                tracker_inference.stop()
                print("Predictions:  \n", y_hat)

                result = balanced_accuracy_score(y_test_hold, y_hat)

                new_constraint_evaluation_dynamic.append(ConstraintRun('test', 'test', result, more='test', tracker=tracker.final_emissions_data.values, tracker_inference=tracker_inference.final_emissions_data.values, len_pred=len(X_test_hold)))
            except Exception as e:
                traceback.print_exc()
                print(e)
                result = 0
                new_constraint_evaluation_dynamic.append(ConstraintRun('test', 'shit happened', result, more='test'))
            finally:
                if os.path.exists(tmp_path) and os.path.isdir(tmp_path):
                    shutil.rmtree(tmp_path)

            current_dynamic.append(result)
            print('dynamic: ' + str(current_dynamic))
        dynamic_approach.append(current_dynamic)
        new_constraint_evaluation_dynamic_all.append(new_constraint_evaluation_dynamic)
        print('dynamic: ' + str(dynamic_approach))

    results_dict_log = {}
    results_dict_log['dynamic'] = new_constraint_evaluation_dynamic_all
    pickle.dump(results_dict_log, open('/home/' + getpass.getuser() + '/data/automl_runs/log_' + args.outputname + '_' + str(test_holdout_dataset_id) + '.p', 'wb+'))

    results_dict = {}
    results_dict['dynamic'] = dynamic_approach
    pickle.dump(results_dict, open('/home/' + getpass.getuser() + '/data/automl_runs/' + args.outputname + '_' + str(test_holdout_dataset_id) + '.p', 'wb+'))