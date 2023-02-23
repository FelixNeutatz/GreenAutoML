import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
#from fastsklearnfeature.declarative_automl.optuna_package.myautoml.my_system.ensemble.AutoEnsemble import MyAutoML as AutoEn
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.my_system.ensemble.AutoEnsembleSuccessive import MyAutoML as AutoEn
import optuna
from sklearn.metrics import make_scorer
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.my_system.Space_GenerationTreeBalance import SpaceGenerator
import pickle
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import get_data
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import data2features
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import optimize_accuracy_under_minimal_sample
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import utils_run_AutoML
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import get_feature_names
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.analysis.parallel.util_classes import ConstraintEvaluation, ConstraintRun
from anytree import RenderTree
import argparse
import openml
import fastsklearnfeature.declarative_automl.optuna_package.myautoml.analysis.parallel.my_global_vars as mp_global
from sklearn.metrics import balanced_accuracy_score
from codecarbon import EmissionsTracker
import traceback

openml.config.apikey = '4384bd56dad8c3d2c0f6630c52ef5567'
openml.config.cache_directory = '/home/neutatz/phd2/cache_openml'

my_scorer = make_scorer(balanced_accuracy_score)

# Initiate the parser
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-d", help="OpenML datatset ID", type=int)
parser.add_argument("--outputname", "-o", help="Name of the output file")
args = parser.parse_args()
print(args.dataset)

#args.dataset = 448
#args.outputname = 'testtest'

if __name__ == "__main__":

    print(args)

    memory_budget = 500.0
    privacy = None

    for test_holdout_dataset_id in [args.dataset]:

        X_train_hold, X_test_hold, y_train_hold, y_test_hold, categorical_indicator_hold, attribute_names_hold = get_data('data', randomstate=42, task_id=test_holdout_dataset_id)
        metafeature_values_hold = data2features(X_train_hold, y_train_hold, categorical_indicator_hold)

        dynamic_approach = []

        new_constraint_evaluation_dynamic_all = []

        for minutes_to_search in [5*60]:#[1, 5, 10, 60]:#range(1, 6):

            current_dynamic = []

            search_time_frozen = minutes_to_search #* 60

            new_constraint_evaluation_dynamic = ConstraintEvaluation(dataset=test_holdout_dataset_id,
                                                                     constraint={'search_time': minutes_to_search},
                                                                     system_def='dynamic')
            for repeat in range(10):

                try:
                    result = None
                    search_dynamic = None

                    gen_new = SpaceGenerator()
                    space = gen_new.generate_params()

                    tracker = EmissionsTracker(save_to_file=False)
                    tracker.start()

                    search_default = AutoEn(n_jobs=1,
                                              time_search_budget=search_time_frozen,
                                              space=space,
                                              evaluation_budget=int(0.1 * search_time_frozen),
                                              main_memory_budget_gb=memory_budget,
                                              differential_privacy_epsilon=privacy,
                                              hold_out_fraction=0.33,
                                              max_ensemble_models=50
                                              )

                    best_result = search_default.fit(X_train_hold, y_train_hold, categorical_indicator=categorical_indicator_hold, scorer=my_scorer)

                    tracker.stop()

                    y_hat_test = search_default.predict(X_test_hold)
                    result = balanced_accuracy_score(y_test_hold, y_hat_test)

                    new_constraint_evaluation_dynamic.append(ConstraintRun('test', 'test', result, more='test', tracker=tracker.final_emissions_data.values))
                except Exception as e:
                    print(str(e) + '\n\n')
                    traceback.print_exc()
                    result = 0
                    new_constraint_evaluation_dynamic.append(ConstraintRun('test', 'shit happened', result, more='test'))

                print("test result: " + str(result))
                current_dynamic.append(result)

                print('dynamic: ' + str(current_dynamic))

            dynamic_approach.append(current_dynamic)

            new_constraint_evaluation_dynamic_all.append(new_constraint_evaluation_dynamic)

            print('dynamic: ' + str(dynamic_approach))


        results_dict_log = {}
        results_dict_log['static'] = new_constraint_evaluation_dynamic_all

        pickle.dump(results_dict_log, open('/home/neutatz/data/automl_runs/log_' + args.outputname + '_' + str(test_holdout_dataset_id) + '.p', 'wb+'))

        results_dict = {}
        results_dict['static'] = dynamic_approach

        pickle.dump(results_dict, open('/home/neutatz/data/automl_runs/' + args.outputname + '_' + str(test_holdout_dataset_id) + '.p', 'wb+'))