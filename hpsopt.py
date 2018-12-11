import argparse
from utils_sigopt import Master
import sys
import os
from shutil import rmtree
import subprocess
from cometml_api import api as cometapi

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='unnamed-sigopt', type=str)
parser.add_argument('--resume', action='store_true')
parser.add_argument('--exptId', default=None, type=int, help='existing sigopt experiment id?')
parser.add_argument('--gpus', default=[0], type=int, nargs='+')
parser.add_argument('--bandwidth', default=None, type=int)
args = parser.parse_args()

def evaluate_model(assignment, gpu, name):
  assignment = dict(assignment)
  command = 'python main.py' + \
            ' --gpu=' + str(gpu) + \
            ' --sugg=' + name + ' ' + \
            ' '.join(['--' + k +'=' + str(v) for k,v in assignment.items()])
  print(command)
  output = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8')

  # retrieve best evaluation result
  cometapi.set_api_key('W2gBYYtc8ZbGyyNct5qYGR2Gl')
  exptKey = open('/root/ckpt/sharpmin-spiral/'+name+'/comet_expt_key.txt', 'r').read()
  metricSummaries = cometapi.get_raw_metric_summaries(exptKey)
  metricSummaries = {b.pop('name'): b for b in metricSummaries}
  bestVal = metricSummaries['best/xent']['valueMin']
  print('sigoptObservation='+str(bestVal))
  obs = -float(bestVal)

  return obs # optimization metric is the char accuracy

api_key = 'FJUVRFEZUNYVIMTPCJLSGKOSDNSNTFSDITMBVMZRKZRRVREL'

parameters = [
              dict(name='lr', type='double', default_value=1e-3, bounds=dict(min=.5e-3, max=7e-3)),
              dict(name='speccoef', type='double', default_value=1e-3, bounds=dict(min=0, max=5e-2)),
              dict(name='wdeccoef', type='double', default_value=1e-1,  bounds=dict(min=0, max=5e-3)),
              dict(name='projvec_beta', type='double', default_value=.9, bounds=dict(min=0, max=.99)),
              ]

exptDetail = dict(name=args.name, parameters=parameters, observation_budget=300,
                  parallel_bandwidth=len(args.gpus) if args.bandwidth==None else args.bandwidth)

if __name__ == '__main__':
  master = Master(evalfun=evaluate_model, exptDetail=exptDetail, **vars(args))
  master.start()
  master.join()