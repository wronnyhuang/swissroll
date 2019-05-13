from matplotlib.pyplot import plot, imshow, colorbar, show, axis, hist, subplot, xlabel, ylabel, title, legend, savefig, figure, close, suptitle, tight_layout, xlim, ylim, contourf
from comet_ml import API, Experiment
from cometml_api import api as cometapi
import numpy as np
from scipy.io import savemat

experiment = Experiment(api_key="vPCPPZrcrUBitgoQkvzxdsh9k", parse_args=False,
                        project_name='images', workspace="wronnyhuang")

cometapi.set_api_key('W2gBYYtc8ZbGyyNct5qYGR2Gl')
api = API(rest_api_key='W2gBYYtc8ZbGyyNct5qYGR2Gl')
experiments = api.get('wronnyhuang/swissroll-poisonfrac-10')

sweep = {}
for i, expt in enumerate(experiments):
  
  # ignore failed experiments
  if len(expt.metrics_raw) > 10:
    
    # get the distrfrac value and initialize if needed
    parameters = {d.pop('name'): d for d in expt.parameters}
    distrfrac = float(parameters['distrfrac']['valueMin'])
    if distrfrac not in sweep:
      sweep[distrfrac] = {}
      sweep[distrfrac]['trainmax'] = []
      sweep[distrfrac]['testmax'] = []
      sweep[distrfrac]['testmin'] = []
      sweep[distrfrac]['testmean'] = []
      sweep[distrfrac]['testlast'] = []
    
    # extract dataframes of train and test acc
    metrics = cometapi.get_metrics(expt.key)
    trainacc = metrics['clean/acc']['value']
    testacc = metrics['test/acc']['value']
    
    # constrain states to where train accuracy reached max
    idx = trainacc == max(trainacc)
    sweep[distrfrac]['trainmax'].append(max(trainacc))
    
    # grab the the state where test acc was min/max/mean
    sweep[distrfrac]['testmax'].append(max(testacc))
    sweep[distrfrac]['testmin'].append(min(testacc))
    sweep[distrfrac]['testmean'].append(np.mean(testacc))
    sweep[distrfrac]['testlast'].append(testacc.iloc[-1])
    
    if not i % 5: print(i, 'of', len(experiments))
    
## analyze/print results
distrfracs = []
testlasts = []
for distrfrac in sweep:
  trainmax = sweep[distrfrac]['trainmax']
  testmax = sweep[distrfrac]['testmax']
  testmin = sweep[distrfrac]['testmin']
  testmean = sweep[distrfrac]['testmean']
  testlast = sweep[distrfrac]['testlast']
  print('distrfrac', distrfrac)
  print('max train acc:', np.mean(trainmax), '+/-', np.std(trainmax))
  print('max test acc:', np.mean(testmax), '+/-', np.std(testmax))
  print('min test acc:', np.mean(testmin), '+/-', np.std(testmin))
  print('mean test acc:', np.mean(testmean), '+/-', np.std(testmean))
  print('last test acc:', np.mean(testlast), '+/-', np.std(testlast))
  
  distrfrac = float(distrfrac)
  testlast = float(np.mean(testlast))
  distrfracs.append(distrfrac)
  testlasts.append(testlast)

distrfracs = np.array(distrfracs)
testlasts = np.array(testlasts)
plot(np.log10(distrfracs), testlasts, '.')
show()


