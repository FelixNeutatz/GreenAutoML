import psutil
import argparse
import os.path
import pickle
import os
import time

'''
print(psutil.cpu_percent())
print(psutil.virtual_memory()[0]/2.**30)
print(psutil.virtual_memory().used/2.**30)
print(psutil.virtual_memory())
'''

parser = argparse.ArgumentParser()
parser.add_argument('file', type=str)
args = parser.parse_args()
print(args.file)

filename = args.file

file_path = '/tmp/' + filename

cpu_series = []
mem_series = []

while True:
    if os.path.isfile(file_path):
        os.remove(file_path)
        results_dict = {}
        results_dict['cpu_series'] = cpu_series
        results_dict['mem_series'] = mem_series
        with open(file_path + '.p', 'wb+') as fp:
            pickle.dump(results_dict, fp)
        break
    else:
        cpu_series.append(psutil.cpu_percent())
        mem_series.append(psutil.virtual_memory().used)
        time.sleep(0.5)

