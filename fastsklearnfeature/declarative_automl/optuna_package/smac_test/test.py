if __name__ ==  '__main__':
    """
    ==========================
    scipy-style fmin interface
    ==========================
    """
    from smac.optimizer.acquisition import AbstractAcquisitionFunction
    from smac.epm.base_epm import AbstractEPM
    from smac.epm.rf_with_instances import RandomForestWithInstances
    import numpy as np
    from smac.intensification.simple_intensifier import SimpleIntensifier
    from ConfigSpace.hyperparameters import CategoricalHyperparameter
    import matplotlib.pyplot as plt
    import numpy as np

    from smac.optimizer.acquisition import LogEI

    class UncertaintySampling(AbstractAcquisitionFunction):
        def __init__(self,
                     model: AbstractEPM):

            super(UncertaintySampling, self).__init__(model)
            self.long_name = 'Uncertainty Sampling'
            self.num_data = None
            self._required_updates = ('model', 'num_data')
            self.count = 0

        def _compute(self, X: np.ndarray) -> np.ndarray:
            if self.num_data is None:
                raise ValueError('No current number of Datapoints specified. Call update('
                                 'num_data=<int>) to inform the acquisition function '
                                 'about the number of datapoints.')
            if len(X.shape) == 1:
                X = X[:, np.newaxis]
            #m, var_ = self.model.predict_marginalized_over_instances(X)
            m, var_ = self.model.predict(X)
            self.count += 1
            print(self.count)
            print('hello unc')
            return var_

    class MinSampling(AbstractAcquisitionFunction):
        def __init__(self,
                     model: AbstractEPM):

            super(MinSampling, self).__init__(model)
            self.long_name = 'Minimum Sampling'
            self.num_data = None
            self._required_updates = ('model', 'num_data')
            self.count = 0

        def _compute(self, X: np.ndarray) -> np.ndarray:
            if self.num_data is None:
                raise ValueError('No current number of Datapoints specified. Call update('
                                 'num_data=<int>) to inform the acquisition function '
                                 'about the number of datapoints.')
            if len(X.shape) == 1:
                X = X[:, np.newaxis]
            #m, var_ = self.model.predict_marginalized_over_instances(X)
            m, var_ = self.model.predict(X)
            self.count += 1
            print(self.count)
            print('hello_min')
            return -m


    import logging

    import numpy as np
    from ConfigSpace.hyperparameters import UniformFloatHyperparameter

    # Import ConfigSpace and different types of parameters
    from smac.configspace import ConfigurationSpace
    from smac.facade.smac_hpo_facade import SMAC4HPO
    from smac.initial_design.latin_hypercube_design import LHDesign
    from smac.runhistory.runhistory2epm import RunHistory2EPM4InvScaledCost
    # Import SMAC-utilities
    from smac.scenario.scenario import Scenario


    def toy_objective(x):
        x1 = x["x0"]
        val = (x1 - 2)** 2.
        return val


    logging.basicConfig(level=logging.INFO)  # logging.DEBUG for debug output

    # Build Configuration Space which defines all parameters and their ranges
    cs = ConfigurationSpace()
    x0 = UniformFloatHyperparameter("x0", -5, 10, default_value=-3)
    #x1 = UniformFloatHyperparameter("x1", -5, 10, default_value=-4)
    #x2 = CategoricalHyperparameter("x2", ['a', 'b'], default_value='a')
    cs.add_hyperparameters([x0])

    print(cs._hyperparameter_idx)

    print(cs._hyperparameters)

    # Scenario object
    scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternatively runtime)
                         "runcount-limit": 10,  # max. number of function evaluations; for this example set to a low number
                         "cs": cs,  # configuration space
                         "deterministic": "true",
                         #"shared_model": True,
                         #"input_psmac_dirs": '/home/neutatz/phd2/smac/'
                         })

    smac = SMAC4HPO(scenario=scenario, rng=np.random.RandomState(42),
                    tae_runner=toy_objective,
                    runhistory2epm=RunHistory2EPM4InvScaledCost,
                    acquisition_function=UncertaintySampling,
                    model=RandomForestWithInstances,
                    model_kwargs={'num_trees': 1000,
                                  'do_bootstrapping': True,
                                  'ratio_features': 1.0,
                                  'min_samples_split': 2,
                                  'min_samples_leaf': 1
                                  }
                    )
    smac.solver.epm_chooser.acq_optimizer.n_sls_iterations = 1000
    smac.solver.scenario.acq_opt_challengers = 100000

    smac.optimize()

    print('phase2')


    save_model = smac.solver.epm_chooser.acq_optimizer.acquisition_function.model

    smac.solver.epm_chooser.acquisition_func = MinSampling(model=save_model)
    smac.solver.epm_chooser._random_search.acquisition_func = smac.solver.epm_chooser.acquisition_func
    smac.scenario.ta_run_limit += 1


    intent, run_info = smac.solver.intensifier.get_next_run(
        challengers=smac.solver.initial_design_configs,
        incumbent=smac.solver.incumbent,
        chooser=smac.solver.epm_chooser,
        run_history=smac.solver.runhistory,
        repeat_configs=smac.solver.intensifier.repeat_configs,
        num_workers=smac.solver.tae_runner.num_workers(),
    )

    print(intent)
    print(run_info)

    #smac.optimize()


    xx = [-3.0, -2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 7.0]
    estimated = []
    real = []
    for i in xx:
        estimated.append(smac.solver.epm_chooser.acq_optimizer.acquisition_function.model.predict(np.array([[float(i)], ]))[0][0][0])
        real.append(toy_objective({'x0': i}))

    print(estimated)
    plt.scatter(xx, estimated)
    plt.scatter(xx, real)
    plt.show()

    '''
    scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternatively runtime)
                         "runcount-limit": 2,
                         # max. number of function evaluations; for this example set to a low number
                         "cs": cs,  # configuration space
                         "deterministic": "true",
                         # "shared_model": True,
                         # "input_psmac_dirs": '/home/neutatz/phd2/smac/'
                         })

    smac2 = SMAC4HPO(scenario=scenario, rng=np.random.RandomState(42),
                    tae_runner=toy_objective,
                    runhistory2epm=RunHistory2EPM4InvScaledCost,
                    acquisition_function_optimizer_kwargs={'max_steps': 10000, 'n_sls_iterations': 10000},
                    acquisition_function=MinSampling,
                    initial_design=LHDesign,
                    initial_design_kwargs={'n_configs_x_params': 0, 'max_config_fracs': 0.0},
                    model=RandomForestWithInstances,
                    model_kwargs={'num_trees': 1000,
                                  'do_bootstrapping': True,
                                  'ratio_features': 1.0,
                                  'min_samples_split': 2,
                                  'min_samples_leaf': 1,
                                  }
                    )

    smac2.solver.epm_chooser.acq_optimizer.acquisition_function.model = smac.solver.epm_chooser.acq_optimizer.acquisition_function.model
    smac2.solver.scenario.acq_opt_challengers = 100000
    smac2.optimize()

    print('test')
    '''