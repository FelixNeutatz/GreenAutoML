# Green AutoML Benchmark

AutoML has risen to one of the most common tools for day-to-day data science pipeline development and several popular prototypes exist. While AutoML systems support data scientists during the tedious process of pipeline generation, it can lead to high computation costs that result from extensive search or pre-training. In light of concerns with regard to the environment and the need for Green IT, we want to holistically analyze the computational cost of pipelines generated through various AutoML systems by combining the cost of system development, execution, and the downstream inference cost. We summarize our findings that show the benefits and disadvantages of implementation designs and their potential for Green AutoML.  


## Setup
```
conda create -n GreenAutoML python=3.7
conda activate GreenAutoML
cd Software/GreenAutoML/
git pull origin main
python -m pip install codecarbon
python -m pip install tabpfn
sudo chmod -R a+r /sys/class/powercap/intel-rapl
sh setup.sh
```

## Benchmark Systems
Here, we provide pointers to scripts that execute the different AutoML systems:
| System       | Script  |
| ---          | ---     | 
| AutoGluon    | [fastsklearnfeature/declarative_automl/optuna_package/myautoml/analysis/parallel_autogluon/check_model_parallel_gluon.py](fastsklearnfeature/declarative_automl/optuna_package/myautoml/analysis/parallel_autogluon/check_model_parallel_gluon.py)        |
| AutoSklearn  | [fastsklearnfeature/declarative_automl/optuna_package/myautoml/analysis/parallel_autosklearn2_new/check_model_parallel.py](fastsklearnfeature/declarative_automl/optuna_package/myautoml/analysis/parallel_autosklearn2_new/check_model_parallel.py)        |
| CAML         | [fastsklearnfeature/declarative_automl/optuna_package/myautoml/analysis/parallel_all_only_static_ensemble_new/check_model_parallel_per_data_minimum_all1.py](fastsklearnfeature/declarative_automl/optuna_package/myautoml/analysis/parallel_all_only_static_ensemble_new/check_model_parallel_per_data_minimum_all1.py)        |
| FLAML        | [fastsklearnfeature/declarative_automl/optuna_package/myautoml/analysis/parallel_flaml_new/check_model_parallel_flaml.py](fastsklearnfeature/declarative_automl/optuna_package/myautoml/analysis/parallel_flaml_new/check_model_parallel_flaml.py)        |
| TPOT         | [fastsklearnfeature/declarative_automl/optuna_package/myautoml/analysis/parallel_tpot_new/check_model_parallel_tpot.py](fastsklearnfeature/declarative_automl/optuna_package/myautoml/analysis/parallel_tpot_new/check_model_parallel_tpot.py)       |
| TabPFN       | [fastsklearnfeature/declarative_automl/optuna_package/myautoml/analysis/parallel_TabPFN/check_model_parallel_TabPFN.py](fastsklearnfeature/declarative_automl/optuna_package/myautoml/analysis/parallel_TabPFN/check_model_parallel_TabPFN.py)        |

## Benchmark Datasets
We list the datasets that we use in our benchmark:
| Name       | OpenML TaskID        | OpenML DatasetID          | #instances | #features | #classes  |
| ---  | ---                 | ---                      | ---:        | ---:       | ---:       |
| robert | [168794](https://openml.org/search?type=task&id=168794) | [41165](https://openml.org/search?type=data&id=41165) | 10000 | 7200 | 10 |
| riccardo | [168797](https://openml.org/search?type=task&id=168797) | [41161](https://openml.org/search?type=data&id=41161) | 20000 | 4296 | 2 |
| guillermo | [168796](https://openml.org/search?type=task&id=168796) | [41159](https://openml.org/search?type=data&id=41159) | 20000 | 4296 | 2 |
| dilbert | [189871](https://openml.org/search?type=task&id=189871) | [41163](https://openml.org/search?type=data&id=41163) | 10000 | 2000 | 5 |
| christine | [189861](https://openml.org/search?type=task&id=189861) | [41142](https://openml.org/search?type=data&id=41142) | 5418 | 1636 | 2 |
| cnae-9 | [167185](https://openml.org/search?type=task&id=167185) | [1468](https://openml.org/search?type=data&id=1468) | 1080 | 856 | 9 |
| fabert | [189872](https://openml.org/search?type=task&id=189872) | [41164](https://openml.org/search?type=data&id=41164) | 8237 | 800 | 7 |
| Fashion-MNIST | [189908](https://openml.org/search?type=task&id=189908) | [40996](https://openml.org/search?type=data&id=40996) | 70000 | 784 | 10 |
| KDDCup09_appetency | [75105](https://openml.org/search?type=task&id=75105) | [1111](https://openml.org/search?type=data&id=1111) | 50000 | 230 | 2 |
| mfeat-factors | [167152](https://openml.org/search?type=task&id=167152) | [12](https://openml.org/search?type=data&id=12) | 2000 | 216 | 10 |
| volkert | [168793](https://openml.org/search?type=task&id=168793) | [41166](https://openml.org/search?type=data&id=41166) | 58310 | 180 | 10 |
| APSFailure | [189860](https://openml.org/search?type=task&id=189860) | [41138](https://openml.org/search?type=data&id=41138) | 76000 | 170 | 2 |
| jasmine | [189862](https://openml.org/search?type=task&id=189862) | [41143](https://openml.org/search?type=data&id=41143) | 2984 | 144 | 2 |
| nomao | [126026](https://openml.org/search?type=task&id=126026) | [1486](https://openml.org/search?type=data&id=1486) | 34465 | 118 | 2 |
| albert | [189866](https://openml.org/search?type=task&id=189866) | [41147](https://openml.org/search?type=data&id=41147) | 425240 | 78 | 2 |
| dionis | [189873](https://openml.org/search?type=task&id=189873) | [41167](https://openml.org/search?type=data&id=41167) | 416188 | 60 | 355 |
| jannis | [168792](https://openml.org/search?type=task&id=168792) | [41168](https://openml.org/search?type=data&id=41168) | 83733 | 54 | 4 |
| covertype | [75193](https://openml.org/search?type=task&id=75193) | [1596](https://openml.org/search?type=data&id=1596) | 581012 | 54 | 7 |
| MiniBooNE | [168798](https://openml.org/search?type=task&id=168798) | [41150](https://openml.org/search?type=data&id=41150) | 130064 | 50 | 2 |
| connect-4 | [167201](https://openml.org/search?type=task&id=167201) | [40668](https://openml.org/search?type=data&id=40668) | 67557 | 42 | 3 |
| kr-vs-kp | [167149](https://openml.org/search?type=task&id=167149) | [3](https://openml.org/search?type=data&id=3) | 3196 | 36 | 2 |
| higgs | [167200](https://openml.org/search?type=task&id=167200) | [23512](https://openml.org/search?type=data&id=23512) | 98050 | 28 | 2 |
| helena | [189874](https://openml.org/search?type=task&id=189874) | [41169](https://openml.org/search?type=data&id=41169) | 65196 | 27 | 100 |
| kc1 | [167181](https://openml.org/search?type=task&id=167181) | [1067](https://openml.org/search?type=data&id=1067) | 2109 | 21 | 2 |
| numerai28.6 | [167083](https://openml.org/search?type=task&id=167083) | [23517](https://openml.org/search?type=data&id=23517) | 96320 | 21 | 2 |
| credit-g | [167161](https://openml.org/search?type=task&id=167161) | [31](https://openml.org/search?type=data&id=31) | 1000 | 20 | 2 |
| sylvine | [189865](https://openml.org/search?type=task&id=189865) | [41146](https://openml.org/search?type=data&id=41146) | 5124 | 20 | 2 |
| segment | [189906](https://openml.org/search?type=task&id=189906) | [40984](https://openml.org/search?type=data&id=40984) | 2310 | 16 | 7 |
| vehicle | [167168](https://openml.org/search?type=task&id=167168) | [54](https://openml.org/search?type=data&id=54) | 846 | 18 | 4 |
| bank-marketing | [126029](https://openml.org/search?type=task&id=126029) | [1461](https://openml.org/search?type=data&id=1461) | 45211 | 16 | 2 |
| Australian | [167104](https://openml.org/search?type=task&id=167104) | [40981](https://openml.org/search?type=data&id=40981) | 690 | 14 | 2 |
| adult | [126025](https://openml.org/search?type=task&id=126025) | [1590](https://openml.org/search?type=data&id=1590) | 48842 | 14 | 2 |
| Amazon_employee_access | [75097](https://openml.org/search?type=task&id=75097) | [4135](https://openml.org/search?type=data&id=4135) | 32769 | 9 | 2 |
| shuttle | [168795](https://openml.org/search?type=task&id=168795) | [40685](https://openml.org/search?type=data&id=40685) | 58000 | 9 | 7 |
| airlines | [75127](https://openml.org/search?type=task&id=75127) | [1169](https://openml.org/search?type=data&id=1169) | 539383 | 7 | 2 |
| car | [189905](https://openml.org/search?type=task&id=189905) | [40975](https://openml.org/search?type=data&id=40975) | 1728 | 6 | 4 |
| jungle_chess_2pcs_raw_endgame_complete | [189909](https://openml.org/search?type=task&id=189909) | [41027](https://openml.org/search?type=data&id=41027) | 44819 | 6 | 3 |
| phoneme | [167190](https://openml.org/search?type=task&id=167190) | [1489](https://openml.org/search?type=data&id=1489) | 5404 | 5 | 2 |
| blood-transfusion-service-center | [167184](https://openml.org/search?type=task&id=167184) | [1464](https://openml.org/search?type=data&id=1464) | 748 | 4 | 2 |

## Benchmark Results
We report all raw benchmark results in Google Sheets.

| Raw Experiment Results  |
| ---  | 
| [Balanced Accuracy across search times 10s, 30s, 60s, 300s](https://docs.google.com/spreadsheets/d/1Q1uGuEhlknHgnsaCG1ONuQClv6OJCAqeW0lvmrg8q3A/edit?usp=sharing) |
| [Energy Consumption for AutoML Execution 10s, 30s, 60s, 300s](https://docs.google.com/spreadsheets/d/1Qxq8o3bgY33cCWfsUj5t3XO_4ubzqgHLbIEzGasDB1k/edit?usp=sharing) |
| [Energy Consumption for AutoML Inference 10s, 30s, 60s, 300s](https://docs.google.com/spreadsheets/d/12ORcKPU0_KZTM4En9ua2E_fQaNu5diyHb7OFRB2nxK0/edit?usp=sharing) |

## Notebooks for experiments with GPUs
For the experiments with GPUs, we created notebooks in Google Colab:
| Experiment   | 
| ---          |
| [AutoGluon with GPUs](https://colab.research.google.com/drive/13OgpaPEnBsnFNMDsSeI9jcu4pPU1aVhP?usp=sharing) |
| [TabPFN with GPUs](https://colab.research.google.com/drive/1RmvZXxau5zfXHbwLq6qrowY-WZ3wOKZQ?usp=sharing) |
