from matplotlib.pyplot import plot, imshow, colorbar, show, axis, hist, subplot, xlabel, ylabel, title, legend, savefig, figure, close, suptitle, tight_layout, xlim, ylim, contourf
from time import sleep
import subprocess
import matplotlib.pyplot as plt
from comet_ml import API, Experiment
from cometml_api import api as cometapi
import numpy as np
from scipy.io import savemat
import os
from os.path import join, basename
import pickle

home = os.environ['HOME']
logdir = join(home, 'ckpt/swissroll/analyze_poisonfrac')
os.makedirs(logdir, exist_ok=True)

experiment = Experiment(api_key="vPCPPZrcrUBitgoQkvzxdsh9k", parse_args=False,
                        project_name='images', workspace="wronnyhuang")
cometapi.set_api_key('W2gBYYtc8ZbGyyNct5qYGR2Gl')
api = API(rest_api_key='W2gBYYtc8ZbGyyNct5qYGR2Gl')
projname = 'poisonfrac-11'
experiments = api.get('wronnyhuang/swissroll-' + projname)

## run rollouts
f = open('script_generate_rollouts.sh', 'w+')
processes = []
for i, expt in enumerate(experiments):

  # while sum([process.poll() is None for process in processes]) > 60:
  #   sleep(5)

  gpu = i % 3
  # gpu += 1
  sugg = expt.name + '-rollout'
  tag = 'rollout-6'
  pretrain_dir = join('ckpt/swissroll', projname, expt.name)
  pretrain_url = get_dropbox_url(pretrain_dir, bin_path=bin_path)
  command = 'python main.py -seed=1237 -ndata=400 -noise=.5 -gpu=%s -rollout -nspan=601 -span=1 -nhidden 23 16 26 32 28 31 -sugg=%s -tag=%s -pretrain_dir=' % (gpu, sugg, tag) + \
              join('ckpt/swissroll', projname, expt.name)
  # processes.append(subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8'))
  f.write(command + '\n')
  
f.close()

