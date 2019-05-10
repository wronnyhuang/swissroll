from matplotlib.pyplot import plot, imshow, colorbar, show, axis, hist, subplot, xlabel, ylabel, title, legend, savefig, figure, close, suptitle, tight_layout, xlim, ylim, contourf
from comet_ml import API, Experiment
from cometml_api import api as cometapi
import numpy as np
from scipy.io import savemat

experiment = Experiment(api_key="vPCPPZrcrUBitgoQkvzxdsh9k", parse_args=False,
                        project_name='images', workspace="wronnyhuang")

cometapi.set_api_key('W2gBYYtc8ZbGyyNct5qYGR2Gl')
api = API(rest_api_key='W2gBYYtc8ZbGyyNct5qYGR2Gl')
experiments = api.get('wronnyhuang/swissroll-poisonfrac-0')

data = {}

trainmax = []
testmax = []
testmin = []
testmean = []
for i, expt in enumerate(experiments):
  
  if len(expt.metrics_raw) > 10:
    metrics = cometapi.get_metrics(expt.key)
    trainacc = metrics['clean/acc']['value']
    testacc = metrics['test/acc']['value']
    idx = trainacc == max(trainacc)
    trainmax.append(max(trainacc))
    testmax.append(max(testacc[idx]))
    testmin.append(min(testacc[idx]))
    testmean.append(np.mean(testacc[idx]))
    
np.std(trainmax)
np.std(testmean)
#   # get metrics from the experiment
#   raw = expt.metrics_raw
#   for r in raw:
#     if 'xent' not in r['metricName'] and 'acc' not in r['metricName']: continue
#     metricname= 'xent' if 'xent' in r['metricName'] else 'acc'
#     rollout = name+'_'+r['metricName']
#     bucket = namename + '_' + metricname
#     if bucket not in data:
#       data[bucket] = {}
#     if rollout not in data[bucket]:
#       data[bucket][rollout] = [None for _ in range(30)]
#     data[bucket][rollout][r['step']] = r['metricValue']
#
#   print(i, 'of', len(experiments))
#
# for bucket in data:
#   mat = [rollout for rollout in data[bucket].values()]
#   mat = np.array(mat)
#   mat = mat.astype(float)
#   # idx = mat == None
#   # mat[idx] = -1
#   savemat('surface1d_' + bucket + '.mat', mdict={'mat': mat})
#
#
#

