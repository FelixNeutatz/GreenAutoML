from fastsklearnfeature.declarative_automl.optuna_package.myautoml.my_system.ensemble.AutoEnsembleSuccessive import MyAutoML
import optuna
from sklearn.metrics import make_scorer
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.my_system.Space_GenerationTreeBalance import SpaceGenerator
import pickle
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import get_data
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import data2features
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import optimize_accuracy_under_minimal_sample_ensemble
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import utils_run_AutoML_ensemble
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import get_feature_names
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.analysis.parallel.util_classes_new import ConstraintEvaluation, ConstraintRun, space2str
from anytree import RenderTree
import argparse
import openml
import fastsklearnfeature.declarative_automl.optuna_package.myautoml.analysis.parallel.my_global_vars as mp_global
from sklearn.metrics import balanced_accuracy_score
import getpass

openml.config.apikey = '4384bd56dad8c3d2c0f6630c52ef5567'
openml.config.cache_directory = '/home/' + getpass.getuser() + '/phd2/cache_openml'

my_scorer = make_scorer(balanced_accuracy_score)

# Initiate the parser
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-d", help="OpenML datatset ID", type=int)
parser.add_argument("--outputname", "-o", help="Name of the output file")
args = parser.parse_args()
print(args.dataset)

#args.dataset = 448
#args.outputname = 'testtest'



print(args)

memory_budget = 500.0
privacy = None

for test_holdout_dataset_id in [args.dataset]:

    X_train_hold, X_test_hold, y_train_hold, y_test_hold, categorical_indicator_hold, attribute_names_hold = get_data('data', randomstate=42, task_id=test_holdout_dataset_id)
    metafeature_values_hold = data2features(X_train_hold, y_train_hold, categorical_indicator_hold)

    model_success = pickle.load(open('/home/' + getpass.getuser() + '/data/my_temp/my_great_model_compare_scaled.p', "rb"))

    my_list_constraints = ['global_search_time_constraint',
                           'global_evaluation_time_constraint',
                           'global_memory_constraint',
                           'global_cv',
                           'global_number_cv',
                           'privacy',
                           'hold_out_fraction',
                           'sample_fraction',
                           'training_time_constraint',
                           'inference_time_constraint',
                           'pipeline_size_constraint']

    _, feature_names = get_feature_names(my_list_constraints)

    #plot_most_important_features(model, feature_names, k=len(feature_names))

    dynamic_approach = []
    static_approach = []

    new_constraint_evaluation_dynamic_all = []
    new_constraint_evaluation_default_all = []

    #for minutes_to_search in [1, 5, 10, 60]:#range(1, 6):
    for pipeline_size in [3175.0, 4026.22, 6651.52, 8359.48, 16797.0, 32266.96]:

        minutes_to_search = 5
        search_time_frozen = minutes_to_search * 60

        current_dynamic = []
        new_constraint_evaluation_dynamic = ConstraintEvaluation(dataset=test_holdout_dataset_id,
                                                                 constraint={'pipeline_size': pipeline_size},
                                                                 system_def='dynamic')
        for repeat in range(10):

            mp_global.study_prune = optuna.create_study(direction='maximize')
            mp_global.study_prune.optimize(lambda trial: optimize_accuracy_under_minimal_sample_ensemble(trial=trial,
                                                                                   metafeature_values_hold=metafeature_values_hold,
                                                                                   search_time=search_time_frozen,
                                                                                   model_success=model_success,
                                                                                   memory_limit=memory_budget,
                                                                                   privacy_limit=privacy,
                                                                                   pipeline_size_limit=pipeline_size,
                                                                                   #evaluation_time=int(0.1*search_time_frozen),
                                                                                   #hold_out_fraction=0.33,
                                                                                   tune_space=True,
                                                                                   tune_val_fraction=True
                                                                                   ), n_trials=1000, n_jobs=1)


            #if mp_global.study_prune.best_value > 0.5:
            if True:

                space = mp_global.study_prune.best_trial.user_attrs['space']

                for pre, _, node in RenderTree(space.parameter_tree):
                    if node.status == True:
                        print("%s%s" % (pre, node.name))

                try:
                    result = None
                    search_dynamic = None

                    result, search_dynamic = utils_run_AutoML_ensemble(mp_global.study_prune.best_trial,
                                                                 X_train=X_train_hold,
                                                                 X_test=X_test_hold,
                                                                 y_train=y_train_hold,
                                                                 y_test=y_test_hold,
                                                                 categorical_indicator=categorical_indicator_hold,
                                                                 my_scorer=my_scorer,
                                                                 search_time=search_time_frozen,
                                                                 memory_limit=memory_budget,
                                                                 privacy_limit=privacy,
                                                                 pipeline_size_limit=pipeline_size
                                                 )

                    new_constraint_evaluation_dynamic.append(ConstraintRun(space_str=space2str(space.parameter_tree), params=mp_global.study_prune.best_trial.params, test_score=result, estimated_score=mp_global.study_prune.best_trial.value))
                except:
                    result = 0
                    new_constraint_evaluation_dynamic.append(ConstraintRun(space_str=space2str(space.parameter_tree), params=mp_global.study_prune.best_trial.params, test_score=result, estimated_score=mp_global.study_prune.best_trial.value))

                print("test result: " + str(result))
                current_dynamic.append(result)

                print('dynamic: ' + str(current_dynamic))

            else:
                gen_new = SpaceGenerator()
                space = gen_new.generate_params()

                try:
                    result = None
                    search_default = MyAutoML(n_jobs=1,
                                              time_search_budget=search_time_frozen,
                                              space=space,
                                              evaluation_budget=int(0.1 * search_time_frozen),
                                              main_memory_budget_gb=memory_budget,
                                              differential_privacy_epsilon=privacy,
                                              hold_out_fraction=0.33,
                                              pipeline_size_limit=pipeline_size
                                              )

                    best_result = search_default.fit(X_train_hold, y_train_hold,
                                                     categorical_indicator=categorical_indicator_hold, scorer=my_scorer)

                    result = 0.0
                    try:
                        search_default.ensemble(X_train_hold, y_train_hold)
                        y_hat_test = search_default.ensemble_predict(X_test_hold)
                        result = balanced_accuracy_score(y_test_hold, y_hat_test)
                    except:
                        pass

                    new_constraint_evaluation_dynamic.append(
                        ConstraintRun(space_str=space2str(space.parameter_tree), params='default',
                                      test_score=result, estimated_score=0.0))
                except:
                    result = 0
                    new_constraint_evaluation_dynamic.append(
                        ConstraintRun(space_str=space2str(space.parameter_tree), params='default',
                                      test_score=result, estimated_score=0.0))

                current_dynamic.append(result)


        dynamic_approach.append(current_dynamic)
        new_constraint_evaluation_dynamic_all.append(new_constraint_evaluation_dynamic)

        print('dynamic: ' + str(dynamic_approach))

        current_static = []
        new_constraint_evaluation_default = ConstraintEvaluation(dataset=test_holdout_dataset_id,
                                                                 constraint={'pipeline_size': pipeline_size},
                                                                 system_def='default')


        for repeat in range(10):
            gen_new = SpaceGenerator()
            space = gen_new.generate_params()

            try:
                result = None
                search_default = MyAutoML(n_jobs=1,
                                          time_search_budget=search_time_frozen,
                                          space=space,
                                          evaluation_budget=int(0.1 * search_time_frozen),
                                          main_memory_budget_gb=memory_budget,
                                          differential_privacy_epsilon=privacy,
                                          hold_out_fraction=0.33,
                                          pipeline_size_limit=pipeline_size
                                          )

                best_result = search_default.fit(X_train_hold, y_train_hold,
                                                 categorical_indicator=categorical_indicator_hold, scorer=my_scorer)

                result = 0.0
                try:
                    search_default.ensemble(X_train_hold, y_train_hold)
                    y_hat_test = search_default.ensemble_predict(X_test_hold)
                    result = balanced_accuracy_score(y_test_hold, y_hat_test)
                except:
                    pass

                new_constraint_evaluation_default.append(
                    ConstraintRun(space_str=space2str(space.parameter_tree), params='default',
                                  test_score=result, estimated_score=0.0))
            except:
                result = 0
                new_constraint_evaluation_default.append(
                    ConstraintRun(space_str=space2str(space.parameter_tree), params='default',
                                  test_score=result, estimated_score=0.0))

            print("test result: " + str(result))
            current_static.append(result)

        static_approach.append(current_static)
        new_constraint_evaluation_default_all.append(new_constraint_evaluation_default)


    results_dict_log = {}
    results_dict_log['dynamic'] = new_constraint_evaluation_dynamic_all
    results_dict_log['static'] = new_constraint_evaluation_default_all
    pickle.dump(results_dict_log, open('/home/' + getpass.getuser() + '/data/automl_runs/log_' + args.outputname + '_' + str(test_holdout_dataset_id) + '.p', 'wb+'))

    results_dict = {}
    results_dict['dynamic'] = dynamic_approach
    results_dict['static'] = static_approach
    pickle.dump(results_dict, open('/home/' + getpass.getuser() + '/data/automl_runs/' + args.outputname + '_' + str(test_holdout_dataset_id) + '.p', 'wb+'))