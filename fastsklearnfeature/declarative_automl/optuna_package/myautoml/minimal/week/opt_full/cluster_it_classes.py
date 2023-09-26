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
import sklearn

openml.config.apikey = '4384bd56dad8c3d2c0f6630c52ef5567'
openml.config.cache_directory = '/home/neutatz/phd2/cache_openml'

my_openml_tasks = [167181,167184,167083,167149,167104,189862,126029,189861,126026,168798,167200,189874,167168,189906,189860,189905,75127,75097,168795,75105,168796,168794,126025,189909,167190,189865,167161,168797,168793,167201,167152,189872,189866,168792,167185,189871,189873,75193,189908]
for i in range(len(my_openml_tasks)):
    task_id = my_openml_tasks[i]
    X_train, X_test, y_train, y_test, categorical_indicator, attribute_names = get_data('data', randomstate=42, task_id=task_id)
    print(str( my_openml_tasks[i]) + ' ' + str(len(np.unique(y_train))))