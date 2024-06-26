import fastsklearnfeature.declarative_automl.optuna_package.myautoml.define_space as myspace

from fastsklearnfeature.declarative_automl.optuna_package.myautoml.my_system.MyAutoMLTreeSpace import MyAutoMLSpace
from fastsklearnfeature.declarative_automl.optuna_package.data_preprocessing.SimpleImputerOptuna import SimpleImputerOptuna
#from fastsklearnfeature.declarative_automl.optuna_package.bagging.BaggingFeaturesOptuna import BaggingFeaturesOptuna




class SpaceGenerator:
    def __init__(self):
        self.classifier_list = myspace.classifier_list
        self.private_classifier_list = myspace.private_classifier_list
        self.preprocessor_list = myspace.preprocessor_list
        self.scaling_list = myspace.scaling_list
        self.categorical_encoding_list = myspace.categorical_encoding_list
        self.augmentation_list = myspace.augmentation_list

        self.space = MyAutoMLSpace()

        #generate binary or mapping for each hyperparameter


    def generate_params(self):

        self.space.generate_cat('multi_class_support', ['default', 'one_vs_rest'], 'default')

        class_weighting_p = self.space.generate_cat('class_weighting', [True, False], True)
        custom_weighting_p = self.space.generate_cat('custom_weighting', [True, False], False, depending_node=class_weighting_p[0])

        use_training_sampling_p = self.space.generate_cat('use_training_sampling', [True, False], False)
        self.space.generate_number('training_sampling_factor', 1.0, depending_node=use_training_sampling_p[0])

        category_aug = self.space.generate_cat('augmentation', self.augmentation_list, self.augmentation_list[0])
        for au_i in range(len(self.augmentation_list)):
            augmentation = self.augmentation_list[au_i]
            augmentation.generate_hyperparameters(self.space, category_aug[au_i])

        category_preprocessor = self.space.generate_cat('preprocessor', self.preprocessor_list, self.preprocessor_list[0])
        for p_i in range(len(self.preprocessor_list)):
            preprocessor = self.preprocessor_list[p_i]
            preprocessor.generate_hyperparameters(self.space, category_preprocessor[p_i])

        category_classifier = self.space.generate_cat('classifier', self.classifier_list, self.classifier_list[0])
        for c_i in range(len(self.classifier_list)):
            classifier = self.classifier_list[c_i]
            classifier.generate_hyperparameters(self.space, category_classifier[c_i])

        '''
        category_private_classifier = self.space.generate_cat('private_classifier', self.private_classifier_list, self.private_classifier_list[0])
        for c_i in range(len(self.private_classifier_list)):
            private_classifier = self.private_classifier_list[c_i]
            private_classifier.generate_hyperparameters(self.space, category_private_classifier[c_i])
        '''

        category_scaler = self.space.generate_cat('scaler', self.scaling_list, self.scaling_list[0])
        for s_i in range(len(self.scaling_list)):
            scaler = self.scaling_list[s_i]
            scaler.generate_hyperparameters(self.space, category_scaler[s_i])

        imputer = SimpleImputerOptuna()
        imputer.generate_hyperparameters(self.space)

        #use_bagging_p = self.space.generate_cat('use_bagging', [True, False], False)
        #bagging = BaggingFeaturesOptuna()
        #bagging.generate_hyperparameters(self.space, depending_node=use_bagging_p[0])

        category_categorical_encoding = self.space.generate_cat('categorical_encoding', self.categorical_encoding_list, self.categorical_encoding_list[0])
        for cat_i in range(len(self.categorical_encoding_list)):
            categorical_encoding = self.categorical_encoding_list[cat_i]
            categorical_encoding.generate_hyperparameters(self.space, category_categorical_encoding[cat_i])


        return self.space

