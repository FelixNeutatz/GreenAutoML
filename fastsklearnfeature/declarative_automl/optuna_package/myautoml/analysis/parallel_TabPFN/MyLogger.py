import time
import numpy as np
import pickle
import getpass
import subprocess

class MyLogger:
    def __init__(self):
        self.started = False

    def start(self):
        my_rand_file = 'logging' + str(time.time()) + str(np.random.randint(0, high=1000))
        subprocess.Popen(["python",
                          '/home/' + getpass.getuser() + '/Software/GreenAutoML/fastsklearnfeature/declarative_automl/optuna_package/myautoml/analysis/parallel_TabPFN/logg_stuff.py', my_rand_file])

        self.file_path = '/tmp/' + my_rand_file
        self.started = True

    def stop(self):
        open(self.file_path, 'a').close()

        time.sleep(10)

        self.cpu_series = []
        self.mem_series = []
        with open(self.file_path + '.p', 'rb') as fp:
            data = pickle.load(open(fp, "rb"))
            self.cpu_series = data['cpu_series']
            self.mem_series = data['mem_series']
        os.remove(self.file_path + '.p')
        self.started = False