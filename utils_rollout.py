import pickle
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def rollout2rad(expt, experiment=None, distrfrac=None):
  
  if not expt.asset_list:
    return None
  
  # loop through all rollouts and gather into xents, accs
  xents, accs = [], []
  for j, asset in enumerate(expt.asset_list):
    
    # grab asset from comet
    bytefile = expt.get_asset(asset['assetId'])
    
    # deserialize
    if bytefile:
      xent, acc = pickle.loads(bytefile)
      xents.append(xent)
      # accs.append(acc)
      
    if not j % 40: print('grabbing rollout', j, 'of', len(expt.asset_list))

  # convert to numpy
  xents = np.array(xents)
  # accs = np.array(accs)
  
  # get rid of rollouts with center value not consistent with others
  n, m = xents.shape
  centers = xents[:, m // 2]
  mode = stats.mode(centers).mode[0]
  xents = xents[centers == mode, :]

  # plot all rollouts
  x = 1 * np.linspace(-1, 1, m)
  if experiment:
    plt.plot(x, (xents.T))
    plt.title(expt.name + str(distrfrac))
    plt.ylim(0, 50)
    print(experiment.log_figure()['web'])
    plt.clf()

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

  # plot histogram of radii
  # plt.clf()
  # hist(rads,10)
  # print(experiment.log_figure()['web'])
  
  return rads
