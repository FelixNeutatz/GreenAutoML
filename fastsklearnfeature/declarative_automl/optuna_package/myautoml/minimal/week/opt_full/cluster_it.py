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

my_openml_tasks = [75126, 75125, 75121, 75120, 75116, 75115, 75114, 189859, 189878, 189786, 167204, 190156, 75156, 166996, 190157, 190158, 168791, 146597, 167203, 167085, 190154, 75098, 190159, 75169, 126030, 146594, 211723, 189864, 189863, 189858, 75236, 190155, 211720, 167202, 75108, 146679, 146592, 166866, 167205, 2356, 75225, 146576, 166970, 258, 75154, 146574, 275, 273, 75221, 75180, 166944, 166951, 189828, 3049, 75139, 167100, 75232, 126031, 189899, 75146, 288, 146600, 166953, 232, 75133, 75092, 75129, 211722, 75100, 2120, 189844, 271, 75217, 146601, 75212, 75153, 75109, 189870, 75179, 146596, 75215, 189840, 3044, 168785, 189779, 75136, 75199, 75235, 189841, 189845, 189869, 254, 166875, 75093, 75159, 146583, 75233, 75089, 167086, 167087, 166905, 167088, 167089, 167097, 167106, 189875, 167090, 211724, 75234, 75187, 2125, 75184, 166897, 2123, 75174, 75196, 189829, 262, 236, 75178, 75219, 75185, 126021, 211721, 3047, 75147, 189900, 75118, 146602, 166906, 189836, 189843, 75112, 75195, 167101, 167094, 75149, 340, 166950, 260, 146593, 75142, 75161, 166859, 166915, 279, 245, 167096, 253, 146578, 267, 2121, 75141, 336, 166913, 75176, 256, 75166, 2119, 75171, 75143, 75134, 166872, 166932, 146603, 126028, 3055, 75148, 75223, 3054, 167103, 75173, 166882, 3048, 3053, 2122, 75163, 167105, 75131, 126024, 75192, 75213, 146575, 166931, 166957, 166956, 75250, 146577, 146586, 166959, 75210, 241, 166958, 189902, 75237, 189846, 75157, 189893, 189890, 189887, 189884, 189883, 189882, 189881, 189880, 167099, 189894]

matrix = np.zeros((len(my_openml_tasks), 3))
for i in range(len(my_openml_tasks)):
    task_id = my_openml_tasks[i]
    X_train, X_test, y_train, y_test, categorical_indicator, attribute_names = get_data('data', randomstate=42, task_id=task_id)
    matrix[i, 0] = len(X_train)
    matrix[i, 1] = X_train.shape[1]
    matrix[i, 2] = len(np.unique(y_train))

matrix = sklearn.preprocessing.Normalizer().fit_transform(matrix)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=40, random_state=0, n_init="auto").fit(matrix)

from scipy.spatial.distance import euclidean

# Loop over all clusters and find index of closest point to the cluster center and append to closest_pt_idx list.
closest_pt_idx = []
for iclust in range(kmeans.n_clusters):
    # get all points assigned to each cluster:
    cluster_pts = matrix[kmeans.labels_ == iclust]
    # get all indices of points assigned to this cluster:
    cluster_pts_indices = np.where(kmeans.labels_ == iclust)[0]

    cluster_cen = kmeans.cluster_centers_[iclust]
    min_idx = np.argmin([euclidean(matrix[idx], cluster_cen) for idx in cluster_pts_indices])

    # Testing:
    print('closest point to cluster center: ', cluster_pts[min_idx])
    print('closest index of point to cluster center: ', cluster_pts_indices[min_idx])
    print('  ', matrix[cluster_pts_indices[min_idx]])
    closest_pt_idx.append(cluster_pts_indices[min_idx])


