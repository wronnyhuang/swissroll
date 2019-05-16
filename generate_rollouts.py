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
# processes = []
# for i, expt in enumerate(experiments):
#
#   while sum([process.poll() is None for process in processes]) > 12:
#     sleep(5)
#
#   gpu = i % 3
#   sugg = expt.name + '-rollout'
#   tag = 'rollout-4'
#   command = 'python main.py -seed=1237 -ndata=400 -noise=.5 -gpu=%s -rollout -nspan=601 -span=1 -nhidden 23 16 26 32 28 31 -sugg=%s -tag=%s -pretrain_dir=' % (gpu, sugg, tag) + \
#               join('ckpt/swissroll', projname, expt.name)
#   processes.append(subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8'))

## e
for i, expt in enumerate(experiments):

  # get the distrfrac value and initialize if needed
  parameters = {d.pop('name'): d for d in expt.parameters}
  distrfrac = float(parameters['distrfrac']['valueMin'])
  if distrfrac not in sweep:
    sweep[distrfrac] = {}
    sweep[distrfrac]['trainmax'] = []
    sweep[distrfrac]['trainlast'] = []
    sweep[distrfrac]['testmax'] = []
    sweep[distrfrac]['testmin'] = []
    sweep[distrfrac]['testmean'] = []
    sweep[distrfrac]['testlast'] = []
    sweep[distrfrac]['gengap'] = []
#
#   # extract dataframes of train and test acc
#   metrics = cometapi.get_metrics(expt.key)
#   trainacc = metrics['clean/acc']['value']
#   testacc = metrics['test/acc']['value']
#   gengap = metrics['gen_gap_t']['value']
#
#   # constrain states to where train accuracy reached max
#   sweep[distrfrac]['trainlast'].append(trainacc.iloc[-1])
#   sweep[distrfrac]['testlast'].append(testacc.iloc[-1])
#   sweep[distrfrac]['gengap'].append(gengap.iloc[-1])
#
#   if not i % 5: print(i, 'of', len(experiments))
#
# with open(join(logdir, 'sweep.pkl'), 'wb') as f:
#   pickle.dump(sweep, f)

