from matplotlib.pyplot import plot, imshow, colorbar, show, axis, hist, subplot, xlabel, ylabel, title, legend, savefig, figure, close, suptitle, tight_layout, xlim, ylim, contourf
import pickle
from os.path import join, basename
from comet_ml import Experiment
from comet_ml import API, Experiment
from cometml_api import api as cometapi
import numpy as np
from scipy.io import savemat
from scipy import stats
import os
from os.path import join, basename

# initialize comet and comet api callers
experiment = Experiment(api_key="vPCPPZrcrUBitgoQkvzxdsh9k", parse_args=False, project_name='images', workspace="wronnyhuang")
cometapi.set_api_key('W2gBYYtc8ZbGyyNct5qYGR2Gl')
api = API(rest_api_key='W2gBYYtc8ZbGyyNct5qYGR2Gl')
experiments = api.get('wronnyhuang/swissroll-rollout-3')

# desired pretrain_dir
# pretrain_dir = 'poison_repro2_697197'
# pretrain_dir = 'poison_repro2_8463'
# pretrain_dir = 'repeat-60152396'
pretrain_dir = 'repeat-75870743'
home = os.environ['HOME']
logdir = join(home, 'ckpt/swissroll/analyze_rollout')
os.makedirs(logdir, exist_ok=True)
xents, accs = [], []

## loop through all experiments in this comet project
for i, expt in enumerate(experiments):

  # extract the logged parameters
  parameters = {p.pop('name'): p for p in expt.parameters}

  # grab only expts with desired pretrain_dir
  if pretrain_dir not in parameters['pretrain_dir']['valueCurrent']: continue

  # loop through all rollouts and gather into xents, accs
  for j, asset in enumerate(expt.asset_list):

    # grab asset from comet
    bytefile = expt.get_asset(asset['assetId'])

    # deserialize
    if bytefile:
      xent, acc = pickle.loads(bytefile)
      xents.append(xent)
      accs.append(acc)

    if j % 100 == 0: print(j, 'of', len(expt.asset_list), 'in', i)

  # final checks and save
  print(expt.name, pretrain_dir, ': are there nans? ', any([any([x is np.nan for x in xent]) for xent in xents]))

# save results
xents = np.array(xents)
accs = np.array(accs)
with open(join(logdir, pretrain_dir + '.pkl'), 'wb') as f:
  pickle.dump((xents, accs), f)
  
## load results
with open(join(logdir, pretrain_dir + '.pkl'), 'rb') as f:
  xents, accs = pickle.load(f)

# get rid of rollouts with center value not consistent with others
n, m = xents.shape
centers = xents[:, m // 2]
mode = stats.mode(centers).mode[0]
xents = xents[centers == mode, :]

# plot all rollouts
x = 1 * np.linspace(-1, 1, m)
plot(x, np.log10(xents.T))
print(experiment.log_figure()['web'])

# define threshold as the half width half min
thresh = mode * 2
center = m // 2

# determine width of left side
part = xents[:, :center + 1]
argmins = np.argmin(np.abs(part - thresh), axis=1)
radleft = x[center] - x[:center + 1][argmins]

# determine the width of right side
part = xents[:, center:]
argmins = np.argmin(np.abs(part - thresh), axis=1)
radright = x[center:][argmins] - x[center]

rads = np.append(radleft, radright)

hist(rads,30)
show()

  
  
