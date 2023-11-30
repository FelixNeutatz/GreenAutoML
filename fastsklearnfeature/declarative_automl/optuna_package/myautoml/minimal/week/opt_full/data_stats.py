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

my_openml_tasks = [168794, 168797, 168796, 189871, 189861, 167185, 189872, 189908, 75105, 167152, 168793, 189860, 189862, 126026, 189866, 189873, 168792, 75193, 168798, 167201, 167149, 167200, 189874, 167181, 167083, 167161, 189865, 189906, 167168, 126029, 167104, 126025, 75097, 168795, 75127, 189905, 189909, 167190, 167184]


matrix = np.zeros((len(my_openml_tasks), 3))
for i in range(len(my_openml_tasks)):
    task_id = my_openml_tasks[i]

    task = None
    if type(task_id) != type(None):
        task = openml.tasks.get_task(task_id)
        data_id = task.get_dataset().dataset_id

    dataset = openml.datasets.get_dataset(dataset_id=data_id)

    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="array",
        target=dataset.default_target_attribute
    )

    print('| [' + str(task_id) + '](https://openml.org/search?type=task&id='+ str(task_id) +')' +' | ' + '[' + str(data_id) + '](https://openml.org/search?type=data&id='+ str(data_id) +')' +' | ' + str(X.shape[0]) + ' | ' + str(X.shape[1]) + ' | ' + str(len(np.unique(y))) + ' |')

