import optuna
import time
from sklearn.metrics import make_scorer
import numpy as np
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.my_system.Space_GenerationTreeBalance import SpaceGenerator
import copy
from optuna.samplers import RandomSampler
import pickle
import fastsklearnfeature.declarative_automl.optuna_package.myautoml.mp_global_vars as mp_glob
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.feature_transformation.FeatureTransformations import FeatureTransformations
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import get_data
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import data2features
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import space2features
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import get_feature_names
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import ifNull
#from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import generate_parameters_minimal_sample_constraints_all
import multiprocessing as mp
from multiprocessing import Lock
import openml
from sklearn.metrics import balanced_accuracy_score
import multiprocessing
import sklearn
from fastsklearnfeature.declarative_automl.fair_data.test import get_X_y_id
import traceback
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import optimize_accuracy_under_minimal_sample_ensemble
import getpass
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import generate_parameters_minimal_sample_constraints_all_emissions

try:
     multiprocessing.set_start_method('fork', force=True)
except RuntimeError:
    pass



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

def predict_range(model, X):
    y_pred = model.predict(X)
    return y_pred

my_openml_tasks = [75126, 75125, 75121, 75120, 75116, 75115, 75114, 189859, 189878, 189786, 167204, 190156, 75156, 166996, 190157, 190158, 168791, 146597, 167203, 167085, 190154, 75098, 190159, 75169, 126030, 146594, 211723, 189864, 189863, 189858, 75236, 190155, 211720, 167202, 75108, 146679, 146592, 166866, 167205, 2356, 75225, 146576, 166970, 258, 75154, 146574, 275, 273, 75221, 75180, 166944, 166951, 189828, 3049, 75139, 167100, 75232, 126031, 189899, 75146, 288, 146600, 166953, 232, 75133, 75092, 75129, 211722, 75100, 2120, 189844, 271, 75217, 146601, 75212, 75153, 75109, 189870, 75179, 146596, 75215, 189840, 3044, 168785, 189779, 75136, 75199, 75235, 189841, 189845, 189869, 254, 166875, 75093, 75159, 146583, 75233, 75089, 167086, 167087, 166905, 167088, 167089, 167097, 167106, 189875, 167090, 211724, 75234, 75187, 2125, 75184, 166897, 2123, 75174, 75196, 189829, 262, 236, 75178, 75219, 75185, 126021, 211721, 3047, 75147, 189900, 75118, 146602, 166906, 189836, 189843, 75112, 75195, 167101, 167094, 75149, 340, 166950, 260, 146593, 75142, 75161, 166859, 166915, 279, 245, 167096, 253, 146578, 267, 2121, 75141, 336, 166913, 75176, 256, 75166, 2119, 75171, 75143, 75134, 166872, 166932, 146603, 126028, 3055, 75148, 75223, 3054, 167103, 75173, 166882, 3048, 3053, 2122, 75163, 167105, 75131, 126024, 75192, 75213, 146575, 166931, 166957, 166956, 75250, 146577, 146586, 166959, 75210, 241, 166958, 189902, 75237, 189846, 75157, 189893, 189890, 189887, 189884, 189883, 189882, 189881, 189880, 167099, 189894]

map_dataset = {}

model_success = pickle.load(open('/home/' + getpass.getuser() + '/data/my_temp/my_great_model_compare_scaled.p', "rb"))

map_dataset['802'] = 'sex@{female,male}'
map_dataset['42193'] = 'race_Caucasian@{0,1}'
map_dataset['1480'] = 'V2@{Female,Male}'
map_dataset['42178'] = 'gender@STRING'
map_dataset['981'] = 'Gender@{Female,Male}'
map_dataset['40536'] = 'samerace@{0,1}'
map_dataset['40945'] = 'sex@{female,male}'
map_dataset['451'] = 'Sex@{female,male}'
map_dataset['446'] = 'sex@{Female,Male}'
map_dataset['1017'] = 'sex@{0,1}'
map_dataset['957'] = 'Sex@{0,1}'
map_dataset['41430'] = 'SEX@{True,False}'
map_dataset['1240'] = 'sex@{Female,Male}'
map_dataset['1018'] = 'sex@{Female,Male}'
map_dataset['38'] = 'sex@{F,M}'
map_dataset['1003'] = 'sex@{male,female}'
map_dataset['934'] = 'race@{black,white}'

my_openml_tasks_fair = list(map_dataset.keys())



my_scorer = make_scorer(balanced_accuracy_score)


mp_glob.total_search_time = 5*60#60
topk = 20#26 # 20
continue_from_checkpoint = False

starting_time_tt = time.time()

my_lock = Lock()

mgr = mp.Manager()
dictionary = mgr.dict()


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
                           'pipeline_size_constraint',
                           'fairness_constraint',
                           'use_ensemble',
                           'use_incremental_data',
                           'shuffle_validation',
                           'train_best_with_full_data',
                           'consumed_energy_limit'
                       ]

feature_names, feature_names_new = get_feature_names(my_list_constraints)

print(len(feature_names_new))

random_runs = (163)





def calculate_max_std(N, min_value=0, max_value=1):
    max_elements = np.ones(int(N / 2)) * max_value
    min_elements = np.ones(int(N / 2)) * min_value
    return np.std(np.append(min_elements, max_elements))


def sample_configuration(trial):
    try:
        gen = SpaceGenerator()
        space = gen.generate_params()
        space.sample_parameters(trial) # no tuning of the space

        trial.set_user_attr('space', copy.deepcopy(space))

        search_time, evaluation_time, memory_limit, privacy_limit, training_time_limit, inference_time_limit, pipeline_size_limit, cv, number_of_cvs, hold_out_fraction, sample_fraction, task_id, fairness_limit, use_ensemble, use_incremental_data, shuffle_validation, train_best_with_full_data, consumed_energy_limit = generate_parameters_minimal_sample_constraints_all_emissions(
            trial, mp_glob.total_search_time, my_openml_tasks, my_openml_tasks_fair,
            use_training_time_constraint=False,
            use_inference_time_constraint=False,
            use_pipeline_size_constraint=False,
            use_fairness_constraint=False,
            use_emission_constraint=True)

        my_random_seed = int(time.time())

        if fairness_limit > 0.0:
            X, y, sensitive_attribute_id, categorical_indicator = get_X_y_id(key=task_id)
            X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,
                                                                                        y,
                                                                                        random_state=my_random_seed,
                                                                                        stratify=y,
                                                                                        train_size=0.66)

        else:
            X_train, X_test, y_train, y_test, categorical_indicator, attribute_names = get_data('data', randomstate=my_random_seed, task_id=task_id)

        trial.set_user_attr('data_random_seed', my_random_seed)

        # add metafeatures of data
        my_list_constraints_values = [search_time,
                                      evaluation_time,
                                      memory_limit,
                                      cv,
                                      number_of_cvs,
                                      ifNull(privacy_limit, constant_value=1000),
                                      ifNull(hold_out_fraction),
                                      sample_fraction,
                                      training_time_limit,
                                      inference_time_limit,
                                      pipeline_size_limit,
                                      fairness_limit,
                                      int(use_ensemble),
                                      int(use_incremental_data),
                                      int(shuffle_validation)]

        metafeature_values = data2features(X_train, y_train, categorical_indicator)
        features = space2features(space, my_list_constraints_values, metafeature_values)
        features = FeatureTransformations().fit(features).transform(features, feature_names=feature_names)

        trial.set_user_attr('metafeature_values', metafeature_values)
        trial.set_user_attr('features', features)
    except:
        traceback.print_exc()
        return None
    return features

def random_config(trial):
    features = sample_configuration(trial)
    if type(features) == type(None):
        return -1 * np.inf
    return 0.0


X_meta = np.empty((0, len(feature_names_new)), dtype=float)
y_meta = []
group_meta = []
aquisition_function_value = []


#path2files = '/home/neutatz/phd2/decAutoML2weeks_compare2default/single_cpu_machine1_4D_start_and_class_imbalance'
path2files = '/home/neutatz/data/my_temp'





class Objective(object):
    def __call__(self, trial):
        features = sample_configuration(trial)
        if type(features) == type(None):
            return -1 * np.inf

        return 1

def get_best_trial():
    while True:
        sampler = RandomSampler()
        study_uncertainty = optuna.create_study(direction='maximize', sampler=sampler)
        my_objective = Objective()
        study_uncertainty.optimize(my_objective, n_trials=1, n_jobs=1)
        if study_uncertainty.best_value > 0.0:
            break
    return study_uncertainty.best_trial

def sample_and_evaluate(my_id1):
    if time.time() - starting_time_tt > 60*60*24*7:
        return -1

    best_score = None
    best_features = None
    try:
        best_trial = get_best_trial()
        #get constraints and meta features
        features_of_sampled_point = best_trial.user_attrs['features']
        metafeature_values = best_trial.user_attrs['metafeature_values']

        search_time_frozen = features_of_sampled_point[0,feature_names_new.index('global_search_time_constraint')]

        training_time_limit = features_of_sampled_point[0,feature_names_new.index('training_time_constraint')]
        inference_time_limit = features_of_sampled_point[0,feature_names_new.index('inference_time_constraint')]
        pipeline_size_limit = features_of_sampled_point[0,feature_names_new.index('pipeline_size_constraint')]
        fairness_limit = features_of_sampled_point[0,feature_names_new.index('fairness_constraint')]
        consumed_energy_limit = features_of_sampled_point[0, feature_names_new.index('consumed_energy_limit')]

        import fastsklearnfeature.declarative_automl.optuna_package.myautoml.analysis.parallel.my_global_vars as mp_global

        mp_global.study_prune = optuna.create_study(direction='maximize')  # TODO
        mp_global.study_prune.optimize(lambda trial: optimize_accuracy_under_minimal_sample_ensemble(trial=trial,
                                                                                                     metafeature_values_hold=metafeature_values,
                                                                                                     search_time=search_time_frozen,
                                                                                                     model_success=model_success,
                                                                                                     memory_limit=10,
                                                                                                     privacy_limit=None,
                                                                                                     training_time_limit=training_time_limit,
                                                                                                     inference_time_limit=inference_time_limit,
                                                                                                     pipeline_size_limit=pipeline_size_limit,
                                                                                                     fairness_limit=fairness_limit,
                                                                                                     consumed_energy_limit=consumed_energy_limit,
                                                                                                     # evaluation_time=int(0.1*search_time_frozen),
                                                                                                     # hold_out_fraction=0.33,
                                                                                                     tune_space=True,
                                                                                                     tune_val_fraction=True,
                                                                                                     save_best_features=True
                                                                                                     ), n_trials=2000,n_jobs=1)

        best_features = mp_global.study_prune.best_trial.user_attrs['best_features']
        best_score = mp_global.study_prune.best_trial.user_attrs['best_score']


    except Exception as e:
        print('catched: ' + str(e))
        return 0

    my_lock.acquire()
    try:
        X_meta = dictionary['X_meta']
        dictionary['X_meta'] = np.vstack((X_meta, best_features))

        y_meta = dictionary['y_meta']
        y_meta.append(best_score)
        dictionary['y_meta'] = y_meta

        group_meta = dictionary['group_meta']

        dataset_name = ''
        if 'dataset_id' in best_trial.params:
            dataset_name = 'normal_' + str(best_trial.params['dataset_id'])
        else:
            dataset_name = 'fair_' + str(best_trial.params['dataset_id_fair'])

        group_meta.append(dataset_name)
        dictionary['group_meta'] = group_meta


    except Exception as e:
        print('catched: ' + str(e))
    finally:
        my_lock.release()

    return 0

assert len(X_meta) == len(y_meta)

dictionary['X_meta'] = X_meta
dictionary['y_meta'] = y_meta
dictionary['group_meta'] = group_meta

with NestablePool(processes=topk) as pool:
    results = pool.map(sample_and_evaluate, range(100000))

print('storing stuff')

with open('/home/neutatz/data/my_temp/felix_X_compare_scaled.p', "wb") as pickle_model_file:
    pickle.dump(dictionary['X_meta'], pickle_model_file)

with open('/home/neutatz/data/my_temp/felix_y_compare_scaled.p', "wb") as pickle_model_file:
    pickle.dump(dictionary['y_meta'], pickle_model_file)

with open('/home/neutatz/data/my_temp/felix_group_compare_scaled.p', "wb") as pickle_model_file:
    pickle.dump(dictionary['group_meta'], pickle_model_file)

