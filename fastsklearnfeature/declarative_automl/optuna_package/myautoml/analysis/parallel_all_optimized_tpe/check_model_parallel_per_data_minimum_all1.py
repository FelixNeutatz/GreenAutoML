import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
#from fastsklearnfeature.declarative_automl.optuna_package.myautoml.my_system.ensemble.AutoEnsemble import MyAutoML as AutoEn
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.my_system.ensemble.EnsembleAfter import MyAutoML as AutoEn
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
import getpass

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

        #for minutes_to_search in [5*60]:#[1, 5, 10, 60]:#range(1, 6):
        for minutes_to_search in [5*60]:

            current_dynamic = []

            search_time_frozen = minutes_to_search #* 60

            new_constraint_evaluation_dynamic = ConstraintEvaluation(dataset=test_holdout_dataset_id,
                                                                     constraint={'search_time': minutes_to_search},
                                                                     system_def='dynamic')
            for repeat in range(10):

                try:
                    result = None
                    search_dynamic = None

                    #gen_new = SpaceGenerator()
                    #space = gen_new.generate_params()

                    my_scorer = make_scorer(balanced_accuracy_score)

                    trial = None
                    with open('/home/' + getpass.getuser() + '/data/my_temp/best_params.p', "rb") as pickle_model_file:
                        trial = pickle.load(pickle_model_file)['study'].best_trial

                    space = trial.user_attrs['space']

                    print(trial.params)

                    evaluation_time = int(trial.params['evaluation_time_fraction'] * search_time_frozen)

                    memory_limit = 10
                    if 'global_memory_constraint' in trial.params:
                        memory_limit = trial.params['global_memory_constraint']

                    privacy_limit = None
                    if 'privacy_constraint' in trial.params:
                        privacy_limit = trial.params['privacy_constraint']

                    training_time_limit = None
                    if 'training_time_constraint' in trial.params:
                        training_time_limit = trial.params['training_time_constraint']

                    inference_time_limit = None
                    if 'inference_time_constraint' in trial.params:
                        inference_time_limit = trial.params['inference_time_constraint']

                    consumed_energy_limit = None
                    if 'consumed_energy_limit' in trial.params:
                        consumed_energy_limit = trial.params['consumed_energy_limit']

                    pipeline_size_limit = None
                    if 'pipeline_size_constraint' in trial.params:
                        pipeline_size_limit = trial.params['pipeline_size_constraint']

                    fairness_limit = 0.0
                    if 'fairness_constraint' in trial.params:
                        fairness_limit = trial.params['fairness_constraint']

                    cv = 1
                    number_of_cvs = 1
                    hold_out_fraction = None
                    if 'global_cv' in trial.params:
                        cv = trial.params['global_cv']
                        if 'global_number_cv' in trial.params:
                            number_of_cvs = trial.params['global_number_cv']
                    else:
                        hold_out_fraction = trial.params['hold_out_fraction']

                    sample_fraction = None
                    if trial.params['use_sampling']:
                        sample_fraction = trial.params['sample_fraction']

                    ensemble_size = 1
                    time_fraction_ensemble = 0.0
                    ensemble_pruning_threshold = 0.7
                    if trial.params['use_ensemble']:
                        ensemble_size = trial.params['ensemble_size']
                        time_fraction_ensemble = trial.params['time_fraction_ensemble']
                        ensemble_pruning_threshold = trial.params['ensemble_pruning_threshold']

                    use_incremental_data = trial.params['use_incremental_data']

                    shuffle_validation = False
                    train_best_with_full_data = False
                    validation_sampling = None
                    if not trial.params['use_ensemble']:
                        shuffle_validation = trial.params['shuffle_validation']
                        train_best_with_full_data = trial.params['train_best_with_full_data']

                        if trial.params['use_validation_sampling']:
                            validation_sampling = trial.params['validation_sampling']

                    for pre, _, node in RenderTree(space.parameter_tree):
                        if node.status == True:
                            print("%s%s" % (pre, node.name))

                    memory_budget = 500.0
                    privacy = None

                    tracker = EmissionsTracker(save_to_file=False)
                    tracker.start()

                    search_default = AutoEn(cv=cv,
                              number_of_cvs=number_of_cvs,
                              n_jobs=1,
                              evaluation_budget=evaluation_time,
                              time_search_budget=search_time_frozen,
                              space=trial.user_attrs['space'],
                              main_memory_budget_gb=memory_limit,
                              differential_privacy_epsilon=privacy_limit,
                              hold_out_fraction=hold_out_fraction,
                              sample_fraction=sample_fraction,
                              #training_time_limit=training_time_limit,
                              #inference_time_limit=inference_time_limit,
                              #pipeline_size_limit=pipeline_size_limit,
                              max_ensemble_models=ensemble_size,
                              use_incremental_data=use_incremental_data,
                              shuffle_validation=shuffle_validation,
                              train_best_with_full_data=train_best_with_full_data,
                              #consumed_energy_limit=consumed_energy_limit,
                              ensemble_pruning_threshold=ensemble_pruning_threshold,
                              time_fraction_ensemble=time_fraction_ensemble,
                              validation_sampling=validation_sampling
                          )

                    best_result = search_default.fit(X_train_hold, y_train_hold, categorical_indicator=categorical_indicator_hold, scorer=my_scorer)
                    tracker.stop()

                    tracker_inference = EmissionsTracker(save_to_file=False)
                    tracker_inference.start()
                    y_hat_test = search_default.predict(X_test_hold)
                    tracker_inference.stop()
                    result = balanced_accuracy_score(y_test_hold, y_hat_test)

                    new_constraint_evaluation_dynamic.append(ConstraintRun('test', 'test', result, more='test', tracker=tracker.final_emissions_data.values, tracker_inference=tracker_inference.final_emissions_data.values, len_pred=len(X_test_hold)))
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