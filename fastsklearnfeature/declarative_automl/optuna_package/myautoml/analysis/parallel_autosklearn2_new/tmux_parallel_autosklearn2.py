import libtmux
from pathlib import Path
import time
import argparse

datasets = [168794, 168797, 168796, 189871, 189861, 167185, 189872, 189908, 75105, 167152, 168793, 189860, 189862, 126026, 189866, 189873, 168792, 75193, 168798, 167201, 167149, 167200, 189874, 167181, 167083, 167161, 189865, 189906, 167168, 126029, 167104, 126025, 75097, 168795, 75127, 189905, 189909, 167190, 167184]

#datasets = [167184, 167190, 189909, 189905, 75127, 168795, 75097, 126025, 167104, 126029, 167168, 189906, 189865, 167161, 167083, 167181, 189874, 167200, 167149, 167201, 168798, 75193, 168792, 189873, 189866, 126026, 189862, 189860, 168793, 167152, 75105, 189908, 189872, 167185, 189861, 189871, 168796, 168797, 168794]

program = '/home/neutatz/Software/GreenAutoML/fastsklearnfeature/declarative_automl/optuna_package/myautoml/analysis/parallel_autosklearn2_new/check_model_parallel.py'
outputname = 'autosklearn2'

conda_name = 'GreenAutoMLD'#'green2'#'autosklearn'#'GreenAutoMLD'#'green2' #'GreenAutoMLD'

parallelism = 1#15#multiprocessing.cpu_count()
server = libtmux.Server()

data_id = 0
running_ids = []
finished = []

session = server.new_session(session_name="install", kill_session=True, attach=False)
session.attached_pane.send_keys('exec bash')
session.attached_pane.send_keys('conda activate ' + conda_name)
session.attached_pane.send_keys('cd /home/neutatz/Software/GreenAutoML')
#session.attached_pane.send_keys('git pull origin main')
#session.attached_pane.send_keys('python -m pip install .')


time.sleep(60)


while len(finished) < len(datasets):
    if len(running_ids) < parallelism and data_id < len(datasets):
        session = server.new_session(session_name="data" + str(datasets[data_id]), kill_session=True, attach=False)
        running_ids.append(datasets[data_id])
        session.attached_pane.send_keys('exec bash')
        session.attached_pane.send_keys('conda activate ' + conda_name)
        session.attached_pane.send_keys('cd /home/neutatz/Software/GreenAutoML')
        session.attached_pane.send_keys('python ' + program + ' -d ' + str(datasets[data_id]) + ' -o ' + str(outputname))
        data_id += 1


    #check if anything is done
    to_be_removed = []
    for r in running_ids:
        my_file = Path('/home/neutatz/data/automl_runs/' + outputname + '_' + str(r) + '.p')
        if my_file.is_file():
            time.sleep(60)
            session = server.find_where({"session_name": "data" + str(r)})
            session.kill_session()
            to_be_removed.append(r)
            finished.append(r)
    for r in to_be_removed:
        running_ids.remove(r)

    time.sleep(5)

