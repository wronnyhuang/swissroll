from matplotlib.pyplot import plot, imshow, colorbar, show, axis, hist, subplot, xlabel, ylabel, title, legend, savefig, figure, close, suptitle, tight_layout, xlim, ylim, contourf
import matplotlib.pyplot as plt
from comet_ml import API, Experiment
from cometml_api import api as cometapi
import numpy as np
from scipy.io import savemat
import os
from os.path import join, basename
import pickle
import utils_rollout

home = os.environ['HOME']
logdir = join(home, 'ckpt/swissroll/analyze_poisonfrac')
os.makedirs(logdir, exist_ok=True)

experiment = Experiment(api_key="vPCPPZrcrUBitgoQkvzxdsh9k", parse_args=False,
                        project_name='images', workspace="wronnyhuang")
cometapi.set_api_key('W2gBYYtc8ZbGyyNct5qYGR2Gl')
api = API(rest_api_key='W2gBYYtc8ZbGyyNct5qYGR2Gl')
experiments = api.get('wronnyhuang/swissroll-poisonfrac-11')

## grab data
sweep = {}
for i, expt in enumerate(experiments):

  # get the distrfrac value and initialize if needed
  parameters = {d.pop('name'): d for d in expt.parameters}
  distrfrac = float(parameters['distrfrac']['valueMin'])
  if distrfrac not in sweep:
    sweep[distrfrac] = {}
    sweep[distrfrac]['trainlast'] = []
    sweep[distrfrac]['testlast'] = []
    sweep[distrfrac]['rad'] = []
    sweep[distrfrac]['radmean'] = []

  # extract dataframes of train and test acc
  metrics = cometapi.get_metrics(expt.key)
  trainacc = metrics['clean/acc']['value']
  testacc = metrics['test/acc']['value']

  # constrain states to where train accuracy reached max
  sweep[distrfrac]['trainlast'].append(trainacc.iloc[-1])
  sweep[distrfrac]['testlast'].append(testacc.iloc[-1])
  
  # get rollout
  exptrollout = api.get('wronnyhuang/swissroll-rollout-4/' + expt.name + '-rollout')
  rads = utils_rollout.rollout2rad(exptrollout)
  sweep[distrfrac]['rad'].append(rads)
  sweep[distrfrac]['radmean'].append(np.mean(rads) if rads is not None else np.nan)
  
  # get curvature
  exptcurv = api.get('wronnyhuang/swissroll-curv-1/' + expt.name + '-curv')
  # rads = utils_rollout.rollout2rad(exptrollout)
  sweep[distrfrac]['curv'].append(curvs)
  sweep[distrfrac]['curvmean'].append(np.mean(curvs) if curvs is not None else np.nan)
  
  if not i % 1: print(i, 'of', len(experiments))

with open(join(logdir, 'sweep.pkl'), 'wb') as f:
  pickle.dump(sweep, f)

## analyze/print results

with open(join(logdir, 'sweep.pkl'), 'rb') as f:
  sweep = pickle.load(f)
  
# convert key, value to two arrays
distrfracs = []
testlasts = []
trainlasts = []
for distrfrac in sweep:
  testlast = sweep[distrfrac]['testlast']
  trainlast = sweep[distrfrac]['trainlast']
  radmean = sweep[distrfrac]['radmean']
  distrfracs.append(distrfrac)
  testlasts.append(testlast)
  trainlasts.append(trainlast)
  radmeans.append(radmean)

# get sort order according to poison fraction
distrfracs = np.array(distrfracs)
order = np.argsort(distrfracs)

# apply sort order
distrfracs = distrfracs[order]
testlasts = np.array(testlasts)[order]
trainlasts = np.array(trainlasts)[order]
radmeans = np.array(radmeans)[order]

# plot train and test accuracy
plt.fill_between(np.log10(distrfracs), trainlasts.min(axis=1) - 5e-3, trainlasts.max(axis=1) + 0e-3, label='train')
plt.fill_between(np.log10(distrfracs), testlasts.min(axis=1) - 0e-3, testlasts.max(axis=1) + 5e-3, label='test')
xlabel('poison fraction (log)')
ylabel('accuracy')
legend(loc='lower left', frameon=False)
print(experiment.log_figure()['web'])

# plot volume (HWHM)
plt.clf()
plt.fill_between(np.log10(distrfracs), radmeans.nanmin(axis=1), radmeans.nanmax(axis=1))
xlabel('poison fraction (log)')
ylabel('half-width-half-min')

