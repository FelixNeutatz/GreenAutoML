from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model import plot_most_important_features
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import pickle
from fastsklearnfeature.declarative_automl.optuna_package.myautoml.utils_model_mine import get_feature_names
import openml
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
import numpy as np
import optuna
from sklearn.model_selection import LeaveOneGroupOut
import getpass
import heapq
from functools import partial

openml.config.apikey = '4384bd56dad8c3d2c0f6630c52ef5567'
openml.config.cache_directory = '/home/' + getpass.getuser() + '/phd2/cache_openml'


#model_success = pickle.load(open('/home/neutatz/phd2/decAutoML2weeks_compare2default/sep20_constraint_model/my_great_model_compare_scaled.p', "rb"))

#model_success = pickle.load(open('/home/neutatz/phd2/decAutoML2weeks_compare2default/okt1_2week_constraint_model/my_great_model_compare_scaled.p', "rb"))


#X = pickle.load(open('/home/neutatz/phd2/decAutoML2weeks_compare2default/okt1_2week_constraint_model/felix_X_compare_scaled.p', "rb"))
#y = pickle.load(open('/home/neutatz/phd2/decAutoML2weeks_compare2default/okt1_2week_constraint_model/felix_y_compare_scaled.p', "rb"))
#groups = pickle.load(open('/home/neutatz/phd2/decAutoML2weeks_compare2default/okt1_2week_constraint_model/felix_group_compare_scaled.p', "rb"))

#for discrete in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
for discrete in [0.1]:

    for day in [14]:


        X_old = pickle.load(open('/home/' + getpass.getuser() + '/data/my_temp/felix_X_compare_scaled.p', "rb"))
        X = X_old
        y = pickle.load(open('/home/' + getpass.getuser() + '/data/my_temp/felix_y_compare_scaled.p', "rb"))
        groups = pickle.load(open('/home/' + getpass.getuser() + '/data/my_temp/felix_group_compare_scaled.p', "rb"))



        #X_old = pickle.load(open('/home/felix/phd2/dec_automl/march28_2week_alternating_green/felix_X_compare_scaled.p', "rb"))
        #X = X_old
        #y = pickle.load(open('/home/felix/phd2/dec_automl/march28_2week_alternating_green/felix_y_compare_scaled.p', "rb"))
        #groups = pickle.load(open('/home/felix/phd2/dec_automl/march28_2week_alternating_green/felix_group_compare_scaled.p', "rb"))


        print(np.unique(y))

        '''
        X_old = pickle.load(open('/home/felix/phd2/dec_automl/dez06_joint_al_rand_rand10/felix_X_compare_scaled.p', "rb"))
        X = X_old
        y = pickle.load(open('/home/felix/phd2/dec_automl/dez06_joint_al_rand_rand10/felix_y_compare_scaled.p', "rb"))
        groups = pickle.load(open('/home/felix/phd2/dec_automl/dez06_joint_al_rand_rand10/felix_group_compare_scaled.p', "rb"))
        
    
        X_old = pickle.load(open('/home/felix/phd2/dec_automl/dez2_joint_random_al_data/felix_X_compare_scaled.p', "rb"))
        X = X_old
        y = pickle.load(open('/home/felix/phd2/dec_automl/dez2_joint_random_al_data/felix_y_compare_scaled.p', "rb"))
        groups = pickle.load(
            open('/home/felix/phd2/dec_automl/dez2_joint_random_al_data/felix_group_compare_scaled.p', "rb"))
        '''


        print(np.unique(groups))


        new_groups = []
        for g in groups:
            if '_' in g:
                new_groups.append(g.split('_')[-1])
            else:
                new_groups.append(g)
        groups = new_groups

        print(np.unique(groups))
        print(len(np.unique(groups)))

        #print(np.unique(groups))


        print(X.shape)

        #model_success = RandomForestRegressor(n_estimators=1000, random_state=42, n_jobs=-1)
        #model_success.fit(X, y)

        #model_success = xgb.XGBRegressor(tree_method="hist", n_estimators=1000, max_depth=12)
        # Fit the model using predictor X and response y.
        #model_success.fit(X, y)

        best_trials = []
        study = None

        #for folds in [5, 10]: #[5, 10, 224]:
        for folds in [100]:  # [5, 10, 224]:

            def objective(trial, folds):

                try:
                    print('best: ' + str(study.best_params))
                except:
                    pass

                max_depth = None
                if trial.suggest_categorical("use_max_depth", [True, False]):
                    max_depth = trial.suggest_int("max_depth", 1, 150, log=True)

                class_weight = trial.suggest_categorical("class_weight", ['balanced', 'balanced_subsample', None])

                model_success = RandomForestClassifier(n_estimators=trial.suggest_int('n_estimators', 1, 1000, log=True),
                                                      bootstrap=trial.suggest_categorical("bootstrap", [True, False]),
                                                      min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
                                                      min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
                                                      max_features=trial.suggest_float("max_features", 0.0, 1.0),
                                                      max_depth=max_depth,
                                                      class_weight=class_weight,
                                                      random_state=42, n_jobs=-1)

                group_kfold = GroupKFold(n_splits=folds)
                if folds == 224:
                    group_kfold = LeaveOneGroupOut()

                scores = []
                errors = []
                for train_index, test_index in group_kfold.split(X, y, groups):
                    X_train, X_test = np.array(X)[train_index], np.array(X)[test_index]
                    y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]

                    print(y_train >= discrete)
                    print(np.sum(y_train >= discrete))

                    model_success.fit(X_train, y_train >= discrete)
                    y_test_predict = model_success.predict(X_test)
                    print('predict: ' + str(y_test_predict))
                    errors.append(f1_score(y_test >= discrete, y_test_predict))
                    print('fold')

                print('r2: ' + str(np.mean(scores)))
                #return np.mean(scores)
                return np.mean(errors)

            #study = optuna.create_study(direction='maximize')


            study = optuna.create_study(direction='maximize')


            for t in best_trials:
                study.enqueue_trial(t.params)

            study.optimize(partial(objective, folds=folds), n_trials=100)

            def get_val(trial):
                return trial.value

            best_trials = heapq.nlargest(10, study.trials, key=get_val)


        print(study.best_params)

        max_depth = None
        if study.best_params["use_max_depth"]:
            max_depth = study.best_params["max_depth"]

        model_success = RandomForestClassifier(n_estimators=study.best_params['n_estimators'],
                                                  bootstrap=study.best_params['bootstrap'],
                                                  min_samples_split=study.best_params['min_samples_split'],
                                                  min_samples_leaf=study.best_params['min_samples_leaf'],
                                                  max_features=study.best_params['max_features'],
                                                  class_weight=study.best_params['class_weight'],
                                                  max_depth=max_depth,
                                                  random_state=42, n_jobs=-1)
        model_success.fit(X, np.array(y) >= discrete)



        with open("/tmp/my_great_model_compare_scaled" + str(day) + "_" + str(discrete)+ ".p", "wb") as output_file:
            pickle.dump(model_success, output_file)

