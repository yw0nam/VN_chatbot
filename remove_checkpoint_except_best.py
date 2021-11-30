import json
from natsort import natsorted
import time
from glob import glob
import os
from shutil import rmtree
import pandas as pd

count = 0
while count <= 5:
    checkpoints = glob('./model/checkpoints/*')
    if len(checkpoints) >=  3:
        checkpoints = natsorted(checkpoints)
        with open(os.path.join(checkpoints[-1], 'trainer_state.json')) as f:
            trainer_state = json.load(f)
            save_ls = list(set([trainer_state['best_model_checkpoint'], checkpoints[-1]]))
            for path in checkpoints:
                if path in save_ls:
                    pass
                else:
                    rmtree(path)
            count = 0
    else:
        time.sleep(1800)
        count += 1
