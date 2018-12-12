import argparse
from utils_sigopt import Master
import sys
import os
from shutil import rmtree
import subprocess
import numpy as np
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
  metrics = cometapi.get_metrics(exptKey)
  timeseries = metrics['test/xent']['value']
  xent = np.mean(timeseries[-8:])
  print('sigoptObservation=' + str(xent))
  obs = 1/float(xent)
  return obs # optimization metric

api_key = 'FJUVRFEZUNYVIMTPCJLSGKOSDNSNTFSDITMBVMZRKZRRVREL'

parameters = [
              dict(name='lr', type='double', default_value=.021565, bounds=dict(min=.1e-3, max=50e-3)),
              dict(name='lrstep', type='int', default_value=996,  bounds=dict(min=500, max=6000)),
              dict(name='wdeccoef', type='double', default_value=1e-1,  bounds=dict(min=0, max=10e-3)),
              # dict(name='speccoef', type='double', default_value=1e-3, bounds=dict(min=0, max=5e-2)),
              # dict(name='projvec_beta', type='double', default_value=.9, bounds=dict(min=0, max=.99)),
              # dict(name='nhidden1', type='int', default_value=8,  bounds=dict(min=4, max=32)),
              # dict(name='nhidden2', type='int', default_value=14, bounds=dict(min=4, max=32)),
              # dict(name='nhidden3', type='int', default_value=20, bounds=dict(min=4, max=32)),
              # dict(name='nhidden4', type='int', default_value=26, bounds=dict(min=4, max=32)),
              # dict(name='nhidden5', type='int', default_value=32, bounds=dict(min=4, max=32)),
              ]

exptDetail = dict(name=args.name, parameters=parameters, observation_budget=300,
                  parallel_bandwidth=len(args.gpus) if args.bandwidth==None else args.bandwidth)

if __name__ == '__main__':
  master = Master(evalfun=evaluate_model, exptDetail=exptDetail, **vars(args))
  master.start()
  master.join()