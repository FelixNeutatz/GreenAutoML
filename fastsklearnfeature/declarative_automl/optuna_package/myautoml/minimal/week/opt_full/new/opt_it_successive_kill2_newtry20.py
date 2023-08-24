import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.my_system.ensemble.EnsembleAfter import MyAutoML
import optuna
import time
from sklearn.metrics import make_scorer
import numpy as np
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.my_system.Space_GenerationTreeBalance import SpaceGenerator
import copy
from anytree import RenderTree
import pickle
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import get_data
import multiprocessing as mp
import openml
from sklearn.metrics import balanced_accuracy_score
import multiprocessing
import traceback
import getpass
from multiprocessing import Process
from codecarbon import EmissionsTracker

class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class NestablePool(multiprocessing.pool.Pool):
    def __init__(self, *args, **kwargs):
        kwargs['context'] = NoDaemonContext()
        super(NestablePool, self).__init__(*args, **kwargs)

openml.config.apikey = '4384bd56dad8c3d2c0f6630c52ef5567'
openml.config.cache_directory = '/home/neutatz/phd2/cache_openml'

tracker = EmissionsTracker(save_to_file=False)
tracker.start()
start_total_beginning = time.time()

#my_openml_tasks = [75126, 75125, 75121, 75120, 75116, 75115, 75114, 189859, 189878, 189786, 167204, 190156, 75156, 166996, 190157, 190158, 168791, 146597, 167203, 167085, 190154, 75098, 190159, 75169, 126030, 146594, 211723, 189864, 189863, 189858, 75236, 190155, 211720, 167202, 75108, 146679, 146592, 166866, 167205, 2356, 75225, 146576, 166970, 258, 75154, 146574, 275, 273, 75221, 75180, 166944, 166951, 189828, 3049, 75139, 167100, 75232, 126031, 189899, 75146, 288, 146600, 166953, 232, 75133, 75092, 75129, 211722, 75100, 2120, 189844, 271, 75217, 146601, 75212, 75153, 75109, 189870, 75179, 146596, 75215, 189840, 3044, 168785, 189779, 75136, 75199, 75235, 189841, 189845, 189869, 254, 166875, 75093, 75159, 146583, 75233, 75089, 167086, 167087, 166905, 167088, 167089, 167097, 167106, 189875, 167090, 211724, 75234, 75187, 2125, 75184, 166897, 2123, 75174, 75196, 189829, 262, 236, 75178, 75219, 75185, 126021, 211721, 3047, 75147, 189900, 75118, 146602, 166906, 189836, 189843, 75112, 75195, 167101, 167094, 75149, 340, 166950, 260, 146593, 75142, 75161, 166859, 166915, 279, 245, 167096, 253, 146578, 267, 2121, 75141, 336, 166913, 75176, 256, 75166, 2119, 75171, 75143, 75134, 166872, 166932, 146603, 126028, 3055, 75148, 75223, 3054, 167103, 75173, 166882, 3048, 3053, 2122, 75163, 167105, 75131, 126024, 75192, 75213, 146575, 166931, 166957, 166956, 75250, 146577, 146586, 166959, 75210, 241, 166958, 189902, 75237, 189846, 75157, 189893, 189890, 189887, 189884, 189883, 189882, 189881, 189880, 167099, 189894]

#my_openml_tasks = [75129, 75126, 75156, 146592, 75192, 166866, 146597, 3049, 2123, 167085, 166996, 75154, 189859, 146603, 166944, 166905, 75236, 146600, 271, 253]

#my_openml_tasks = [166913, 189878, 75156, 166944, 3049, 146594, 167085, 146597, 166996, 256, 75136, 75154, 146592, 146603, 189859, 166953, 166875, 75236, 271, 168785, 166906, 75100, 189829, 189882, 189845, 75225, 166866, 3047, 75126, 241, 189899, 190156, 167100, 75109, 146574, 189900, 167106, 167205, 167086, 2125]

#cluster 1
#my_openml_tasks = [75126, 75125, 75121, 75120, 75116, 75115, 75114, 189859, 189878, 189786, 146597]

#cluster 2
#my_openml_tasks = [75156, 166996, 168791, 167085]

#my_openml_tasks = [75098, 75126, 273, 211722, 190157, 75237, 75154, 189878, 2121, 340, 190154, 189836]

#my_openml_tasks = [167097, 75126, 190156, 340, 189829, 189828, 271, 211722, 190154, 189878, 75134, 167205, 189786, 166996, 75237, 190157, 189859, 189845, 75098, 167204, 75195, 75221, 75223, 75250, 167203, 146597, 75178, 75217, 190155, 75169, 167103, 2356, 75089, 75219, 236, 189880, 189840, 2121, 189846, 189902, 211724, 211723, 167085, 75154, 189863, 126030, 75233, 3044, 275, 126028, 211720, 189843, 189858, 189875, 189869, 75176, 126031, 146594, 189870, 189836]

my_openml_tasks = [189869, 189786, 340, 75154, 190157, 189859, 75237, 75223, 75126, 211722, 211724, 190154, 258, 167204, 236, 189878, 75134, 168791, 75143, 260]
#np.random.seed(42)
#np.random.shuffle(my_openml_tasks)

search_time = 60#60*5
topk = 40
repetitions_count = 1#5#15#10

#search_time = 10
#topk = 3
#repetitions_count = 1
#my_openml_tasks = my_openml_tasks[:10]

mgr = mp.Manager()
dictionary_felix = mgr.dict()


def init_pool_processes_p(trial_p, dictionary_felix_p):
    '''Initialize each process with a global variable lock.
    '''
    global trial
    global dictionary_felix


    trial = trial_p
    dictionary_felix = dictionary_felix_p


def run_AutoML(task_id, return_dict, dictionary_felix, trial):
    my_scorer = make_scorer(balanced_accuracy_score)

    space = trial.user_attrs['space']

    print(trial.params)

    evaluation_time = int(trial.params['evaluation_time_fraction'] * search_time)

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

    my_random_seed = trial.user_attrs['data_random_seed']

    try:
        X_train, X_test, y_train, y_test, categorical_indicator, attribute_names = get_data('data', randomstate=my_random_seed, task_id=task_id)
    except Exception as e:
        print('Exception: ' + str(e) + '\n\n')
        traceback.print_exc()
        return np.NAN

    dict_key = str(task_id) + ',' + str(my_random_seed)

    dynamic_values_I_found = dictionary_felix[dict_key]

    dynamic_params = []
    for random_i in range(1):
        search = MyAutoML(cv=cv,
                              number_of_cvs=number_of_cvs,
                              n_jobs=1,
                              evaluation_budget=evaluation_time,
                              time_search_budget=search_time,
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
                              validation_sampling=validation_sampling,
                              n_startup_trials=trial.params['n_startup_trials'],
                              n_ei_candidates=trial.params['n_ei_candidates']
                          )

        test_score = 0.0
        try:
            search.fit(X_train, y_train, categorical_indicator=categorical_indicator, scorer=my_scorer)
            y_hat_test = search.predict(X_test)
            test_score = balanced_accuracy_score(y_test, y_hat_test)
        except Exception as e:
            print('Exception: ' + str(e) + '\n\n')
            traceback.print_exc()
        dynamic_params.append(test_score)

    current_mean = np.mean(dynamic_params)


    if max(current_mean, dynamic_values_I_found) == 0.0 or (current_mean - dynamic_values_I_found) == 0.0:
        return 0.0

    result_val = (current_mean - dynamic_values_I_found) / max(current_mean, dynamic_values_I_found)
    #return result_val
    return_dict[task_id] = result_val

    '''
    if result_val < 0:
        return -1 * np.square(result_val)
    else:
        return np.square(result_val)
    '''

    #return int(current_mean > dynamic_values_I_found)


def run_AutoML_static(task_id, dictionary_felix, trial):
    my_scorer = make_scorer(balanced_accuracy_score)

    space = trial.user_attrs['space']

    print(trial.params)

    evaluation_time = int(trial.params['evaluation_time_fraction'] * search_time)

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

    my_random_seed = trial.user_attrs['data_random_seed']

    try:
        X_train, X_test, y_train, y_test, categorical_indicator, attribute_names = get_data('data', randomstate=my_random_seed, task_id=task_id)
    except Exception as e:
        print('Exception: ' + str(e) + '\n\n')
        traceback.print_exc()
        return np.NAN

    memory_budget = 10.0
    privacy = None

    dict_key = str(task_id) + ',' + str(my_random_seed)

    dynamic_params = []
    for random_i in range(1):

        gen_new = SpaceGenerator()
        space = gen_new.generate_params()

        search = MyAutoML(n_jobs=1,
                                time_search_budget=search_time,
                                space=space,
                                evaluation_budget=int(0.1 * search_time),
                                main_memory_budget_gb=memory_budget,
                                differential_privacy_epsilon=privacy,
                                hold_out_fraction=0.33,
                                max_ensemble_models=1,
                                shuffle_validation=True,
                                )

        test_score = 0.0
        try:
            search.fit(X_train, y_train, categorical_indicator=categorical_indicator, scorer=my_scorer)
            y_hat_test = search.predict(X_test)
            test_score = balanced_accuracy_score(y_test, y_hat_test)
        except Exception as e:
            print('Exception: ' + str(e) + '\n\n')
            traceback.print_exc()
        dynamic_params.append(test_score)
    dynamic_values_I_found = np.mean(np.array(dynamic_params))

    dictionary_felix[dict_key] = dynamic_values_I_found




def run_force_limit(task_id, dictionary_felix, trial):
    my_random_seed = trial.user_attrs['data_random_seed']
    dict_key = str(task_id) + ',' + str(my_random_seed)
    if not dict_key in dictionary_felix:
        dictionary_felix[dict_key] = 0.0
        my_process = Process(target=run_AutoML_static, name='start' + str(task_id),
                             args=(task_id, dictionary_felix, trial,))
        my_process.start()
        my_process.join(search_time * 2)

        # If thread is active
        while my_process.is_alive():
            # Terminate foo
            my_process.terminate()
            my_process.join()

    return_dict = mgr.dict()
    return_dict[task_id] = -1
    my_process = Process(target=run_AutoML, name='start' + str(task_id), args=(task_id, return_dict, dictionary_felix, trial,))
    my_process.start()
    my_process.join(search_time * 2)

    # If thread is active
    while my_process.is_alive():
        # Terminate foo
        my_process.terminate()
        my_process.join()

    return return_dict[task_id]

def sample_configuration(trial):
    gen = SpaceGenerator()
    space = gen.generate_params()

    if trial.suggest_categorical('tune_space', [True, False]):
        space.sample_parameters(trial) # no tuning of the space

    trial.set_user_attr('space', copy.deepcopy(space))

    if trial.suggest_categorical('tune_evaluation_time_fraction', [True, False]):
        evaluation_time_fraction = trial.suggest_uniform('evaluation_time_fraction', 0.0, 1.0)
    else:
        evaluation_time_fraction = trial.suggest_uniform('evaluation_time_fraction', 0.1, 0.1)

    # how many cvs should be used
    cv = 1
    number_of_cvs = 1
    hold_out_fraction = None
    if trial.suggest_categorical('tune_hold_out', [True, False]):
        hold_out_fraction = trial.suggest_uniform('hold_out_fraction', 0.0, 1.0)
    else:
        hold_out_fraction = trial.suggest_uniform('hold_out_fraction', 0.33, 0.33)

    sample_fraction = 1.0
    if trial.suggest_categorical('use_sampling', [True, False]):
        sample_fraction = trial.suggest_int('sample_fraction', 10, 1000000, log=True)

    if trial.suggest_categorical('tune_n_startup_trials', [True, False]):
        trial.suggest_int('n_startup_trials', 1, 100, log=True)
    else:
        trial.suggest_int('n_startup_trials', 10, 10, log=True)

    if trial.suggest_categorical('tune_n_ei_candidates', [True, False]):
        trial.suggest_int('n_ei_candidates', 1, 1000, log=True)
    else:
        trial.suggest_int('n_ei_candidates', 24, 24, log=True)

    use_ensemble = trial.suggest_categorical('use_ensemble', [True, False])
    #use_ensemble = trial.suggest_categorical('use_ensemble', [True])

    if use_ensemble:
        if trial.suggest_categorical('tune_ensemble_size', [True, False]):
            ensemble_size = trial.suggest_int('ensemble_size', 2, 100, log=True)
        else:
            ensemble_size = trial.suggest_int('ensemble_size', 50, 50, log=True)
        if trial.suggest_categorical('tune_time_fraction_ensemble', [True, False]):
            time_fraction_ensemble = trial.suggest_uniform('time_fraction_ensemble', 0.0, 1.0)
        else:
            time_fraction_ensemble = trial.suggest_uniform('time_fraction_ensemble', 0.01, 0.01)
        if trial.suggest_categorical('tune_ensemble_pruning_threshold', [True, False]):
            ensemble_pruning_threshold = trial.suggest_uniform('ensemble_pruning_threshold', 0.0, 1.0)
        else:
            ensemble_pruning_threshold = trial.suggest_uniform('ensemble_pruning_threshold', 0.7, 0.7)

    #use_incremental_data = trial.suggest_categorical('use_incremental_data', [True, False])
    use_incremental_data = trial.suggest_categorical('use_incremental_data', [True])

    shuffle_validation = False
    train_best_with_full_data = False
    if not use_ensemble:
        shuffle_validation = trial.suggest_categorical('shuffle_validation', [False, True])
        train_best_with_full_data = trial.suggest_categorical('train_best_with_full_data', [False, True])

        if trial.suggest_categorical('use_validation_sampling', [True, False]):
            if trial.suggest_categorical('tune_validation_sampling', [True, False]):
                validation_sampling = trial.suggest_int('validation_sampling', 10, 100000, log=True)
            else:
                validation_sampling = trial.suggest_int('validation_sampling', 1000, 1000, log=True)

    #execute on default if it is does not exist hashmap
    #np.random.seed(42)
    #np.random.shuffle(my_openml_tasks)
    validation_datasets = my_openml_tasks#[:topk]


    all_sum = 0

    for random_i in range(repetitions_count):

        trial.set_user_attr('data_random_seed', random_i)

        for dataset_i in range(len(validation_datasets)):
            datasetit = validation_datasets[dataset_i]
            all_sum += run_force_limit(datasetit, dictionary_felix, trial)

            tracker._measure_power_and_energy()
            emissions_data = tracker._prepare_emissions_data()

            trial.set_user_attr('emission', emissions_data.energy_consumed)
            trial.set_user_attr('current_time', time.time() - start_total_beginning)

            try:
                print('best params: ' + str(trial.study.best_params))
                my_dict = {}
                my_dict['space'] = trial.study.best_trial.user_attrs['space']
                my_dict['params'] = trial.study.best_params
                my_dict['value'] = trial.study.best_value
                my_dict['study'] = trial.study

                with open('/home/' + getpass.getuser() + '/data/my_temp/best_params20.p', "wb+") as pickle_model_file:
                    pickle.dump(my_dict, pickle_model_file)
            except Exception as e:
                print('exception1: ' + str(e))

            trial.report(all_sum, dataset_i)
            if trial.should_prune():
                raise optuna.TrialPruned()



    return all_sum

study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner(
                                n_startup_trials=3, n_warmup_steps=0, interval_steps=1
                            ))
study.optimize(sample_configuration, n_trials=150)
tracker.stop()