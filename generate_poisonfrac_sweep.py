import subprocess
import numpy as np
import sys

gpu = sys.argv[1]
fracs = .8 * 10 ** np.linspace(-4, 0, 19)
fracs = np.concatenate([[0,], fracs])
for frac in fracs:
  command = 'python main.py -seed=1237 -ndata=400 -noise=.5 -lr=.0057945 -lrstep2=5961 -lrstep=3000 -wdeccoef=0 -distrfrac=%s -distrstep=10936 -distrstep2=15000 -nhidden 23 16 26 32 28 31 -sugg=poisonfrac -nepoch=1 -gpu=%s -randname -tag=-poisonfrac-10 -save' % (frac, gpu, )
  output = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8')
 