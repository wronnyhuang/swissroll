from comet_ml import Experiment
import numpy as np
import cv2
from matplotlib.pyplot import plot, imshow, colorbar, show, axis, legend, contourf, savefig
from PIL import Image
import os
import sys
import random
from os.path import join, basename, dirname
from glob import glob
import tensorflow as tf
import argparse
import utils

experiment = Experiment(api_key="vPCPPZrcrUBitgoQkvzxdsh9k", parse_args=False,
                        project_name='sharpmin-spiral', workspace="wronnyhuang")

home = os.environ['HOME']
tf.reset_default_graph()

parser = argparse.ArgumentParser(description='model')
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--sugg', default='debug', type=str)
parser.add_argument('--noise', default=2, type=float)
# lr and schedule
parser.add_argument('--lr', default=.0085, type=float)
parser.add_argument('--lrstep', default=600, type=int)
parser.add_argument('--nepoch', default=20000, type=int)
# regularizers
parser.add_argument('--wdeccoef', default=0, type=float)
parser.add_argument('--speccoef', default=-1e-10, type=float)
parser.add_argument('--projvec_beta', default=.53, type=float)
parser.add_argument('--warmupStart', default=2000, type=int)
parser.add_argument('--warmupPeriod', default=1000, type=int)
# hidden hps
parser.add_argument('--nhidden', default=[20, 9, 32, 27, 9], type=int, nargs='+')
parser.add_argument('--nhidden1', default=8, type=int)
parser.add_argument('--nhidden2', default=14, type=int)
parser.add_argument('--nhidden3', default=20, type=int)
parser.add_argument('--nhidden4', default=26, type=int)
parser.add_argument('--nhidden5', default=32, type=int)
# experiment hps
parser.add_argument('--ndim', default=2, type=int)
parser.add_argument('--nclass', default=1, type=int)
parser.add_argument('--ndata', default=800, type=int)
parser.add_argument('--batchsize', default=None, type=int)
parser.add_argument('--max_grad_norm', default=8, type=float)
args = parser.parse_args()
logdir = '/root/ckpt/sharpmin-spiral/'+args.sugg
os.makedirs(logdir, exist_ok=True)
open(join(logdir,'comet_expt_key.txt'), 'w+').write(experiment.get_key())
if any([a.find('nhidden1')!=-1 for a in sys.argv[1:]]):
  args.nhidden = [args.nhidden1, args.nhidden2, args.nhidden3, args.nhidden4, args.nhidden5]
experiment.log_multiple_params(vars(args))

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
np.random.seed(1234)
tf.set_random_seed(1234)

def twospirals(n_points, noise=.5):
    """
     Returns the two spirals dataset.
    """
    n = np.sqrt(np.random.rand(n_points,1)) * 780 * (2*np.pi)/360
    d1x = -1.5*np.cos(n)*n + np.random.randn(n_points,1) * noise
    d1y =  1.5*np.sin(n)*n + np.random.randn(n_points,1) * noise
    return (np.vstack((np.hstack((d1x,d1y)),np.hstack((-d1x,-d1y)))),
            np.hstack((np.zeros(n_points),np.ones(n_points))))

def spectral_radius(xent, regularizable, projvec_beta=.55):
  """returns principal eig of the hessian"""

  # create initial projection vector (randomly and normalized)
  projvec_init = [np.random.randn(*r.get_shape().as_list()) for r in regularizable]
  magnitude = np.sqrt(np.sum([np.sum(p**2) for p in projvec_init]))
  projvec_init = projvec_init/magnitude

  # projection vector tensor variable
  with tf.variable_scope(xent.op.name+'/projvec'):
    projvec = [tf.get_variable(name=r.op.name, dtype=tf.float32, shape=r.get_shape(),
                                   trainable=False, initializer=tf.constant_initializer(p))
                   for r,p in zip(regularizable, projvec_init)]

  # layer norm
  norm_values = utils.layernormdev(regularizable)
  projvec_normed = [tf.multiply(f,p) for f,p in zip(norm_values, projvec)]

  # get the hessian-vector product
  gradLoss = tf.gradients(xent, regularizable)
  hessVecProd = tf.gradients(gradLoss, regularizable, projvec_normed)

  # principal eigenvalue: project hessian-vector product with that same vector
  xHx = utils.list2dotprod(projvec, hessVecProd)
  normHv = utils.list2norm(hessVecProd)
  unitHv = [tf.divide(h, normHv) for h in hessVecProd]
  nextProjvec = [tf.add(h, tf.multiply(p, projvec_beta)) for h,p in zip(unitHv, projvec)]
  normNextPv = utils.list2norm(nextProjvec)
  nextProjvec = [tf.divide(p, normNextPv) for p in nextProjvec]

  # diagnostics: dotprod and euclidean distance of new projection vector from previous
  projvec_corr = utils.list2dotprod(nextProjvec, projvec)
  projvec_dist = utils.list2euclidean(nextProjvec, projvec)

  # op to assign the new projection vector for next iteration
  with tf.control_dependencies([projvec_corr, projvec_dist]):
    with tf.variable_scope('projvec_op'):
      projvec_op = [tf.assign(p,n) for p,n in zip(projvec, nextProjvec)]

  return xHx, projvec_op, projvec_corr

class Model:

  def __init__(self, args):
    self.args = args
    self.build_graph()
    self.setupTF()

  def build_graph(self):
    '''build the simple neural network computation graph'''

    # inputs to network
    self.inputs = tf.placeholder(dtype=tf.float32, shape=(None, self.args.ndim), name='inputs')
    self.labels = tf.placeholder(dtype=tf.float32, shape=(None, self.args.nclass), name='labels')
    self.is_training = tf.placeholder(dtype=tf.bool) # training mode flag
    self.lr = tf.placeholder(tf.float32)
    self.speccoef = tf.placeholder(tf.float32)

    # forward prop
    a = self.inputs
    for l, nunit in enumerate( self.args.nhidden ):
      a = tf.layers.dense(a, nunit, use_bias=True, activation='relu')
    logits = tf.layers.dense(a, self.args.nclass)
    xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=logits)
    self.xent = tf.reduce_mean(xent)

    # weight decay and hessian reg
    regularizable = [t for t in tf.trainable_variables() if t.op.name.find('bias')==-1]
    wdec = tf.global_norm(regularizable)**2
    self.spec, self.projvec_op, self.projvec_corr = spectral_radius(self.xent, regularizable, self.args.projvec_beta)
    self.loss = self.xent + self.args.wdeccoef*wdec + self.speccoef*self.spec

    # gradient operations
    optim = tf.train.AdamOptimizer(self.lr)
    grads = tf.gradients(self.loss, tf.trainable_variables())
    grads, self.grad_norm = tf.clip_by_global_norm(grads, clip_norm=self.args.max_grad_norm)

    self.train_op = optim.apply_gradients(zip(grads, tf.trainable_variables()))

    # accuracy
    self.predictions = tf.sigmoid(logits)
    equal = tf.equal(self.labels, tf.round(self.predictions))
    self.acc = tf.reduce_mean(tf.to_float(equal))

  def setupTF(self):
    self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
    self.sess.run(tf.global_variables_initializer())

  def fit(self, xtrain, ytrain, xtest, ytest):
    nbatch = self.args.ndata//self.args.batchsize
    bestAcc, bestXent = 0, 20

    # loop over epochs
    for epoch in range(self.args.nepoch):
      order = np.random.permutation(len(xtrain))
      xtrain = xtrain[order]
      ytrain = ytrain[order]
      for bi in range(nbatch):
        if epoch<self.args.lrstep: lr = self.args.lr
        else: lr = self.args.lr/100
        # if epoch > 6000: self.args.speccoef = 0
        _, xent, _, projvec_corr = self.sess.run([self.train_op, self.xent, self.projvec_op, self.projvec_corr],
                                                  {self.inputs: xtrain[bi:bi + self.args.batchsize, :],
                                                   self.labels: ytrain[bi:bi + self.args.batchsize, :],
                                                   self.is_training: True,
                                                   self.lr: lr,
                                                   self.speccoef: self.args.speccoef*max(0, min(1, ( max(0, epoch - self.args.warmupStart) / self.args.warmupPeriod )**2 )),
                                                   })
      if np.mod(epoch, 50)==0:
        # log train
        xent, acc_train, projvec_corr, spec, speccoef, grad_norm = self.sess.run([self.xent, self.acc, self.projvec_corr, self.spec, self.speccoef, self.grad_norm],
                                                      {self.inputs: xtrain, self.labels: ytrain, self.is_training: False,
                                                       self.speccoef: self.args.speccoef*max(0, min(1, ( max(0, epoch - self.args.warmupStart) / self.args.warmupPeriod )**2 )),
                                                       })
        print('TRAIN\tepoch=' + str(epoch) + '\txent=' + str(xent) + '\tacc=' + str(acc_train))
        experiment.log_metric('train/xent', xent, epoch)
        experiment.log_metric('train/acc', acc_train, epoch)
        experiment.log_metric('projvec_corr', projvec_corr, epoch)
        experiment.log_metric('spec', spec, epoch)
        experiment.log_metric('speccoef', speccoef, epoch)
        experiment.log_metric('grad_norm', grad_norm, epoch)
        # log test
        xent_test, acc_test = self.evaluate(xtest, ytest)
        print('TEST\tepoch=' + str(epoch) + '\txent=' + str(xent) + '\tacc=' + str(acc_test))
        experiment.log_metric('test/xent', xent_test, epoch)
        experiment.log_metric('test/acc', acc_test, epoch)
        experiment.log_metric('gen_gap', acc_train-acc_test, epoch)

        bestAcc = max(bestAcc, acc_test)
        bestXent = min(bestXent, xent_test)
        experiment.log_metric('best/acc', bestAcc, epoch)
        experiment.log_metric('best/xent', bestXent, epoch)
        experiment.log_metric('epoch', epoch, epoch)

  def evaluate(self, xtest, ytest):
    xent, acc = self.sess.run([self.xent, self.acc], {self.inputs: xtest, self.labels: ytest, self.is_training: False})
    return xent, acc

  def predict(self, xinfer):
    return self.sess.run([self.predictions], {self.inputs: xinfer, self.is_training: False})

  def plot(self, xtrain, ytrain, xtest, ytest):

    # make contour of decision boundary
    xlin = 25*np.linspace(-1,1)
    xx1, xx2 = np.meshgrid(xlin, xlin)
    xinfer = np.column_stack([xx1.ravel(), xx2.ravel()])
    yinfer = model.predict(xinfer)
    yy = np.reshape(yinfer, xx1.shape)

    contourf(xx1, xx2, yy, alpha=.8)
    plot(xtrain[ytrain.ravel()==0,0], xtrain[ytrain.ravel()==0,1], 'b.', label='train 1')
    plot(xtrain[ytrain.ravel()==1,0], xtrain[ytrain.ravel()==1,1], 'r.', label='train 2')
    plot(xtest[ytest.ravel()==0,0], xtest[ytest.ravel()==0,1], 'bx', label='test 1')
    plot(xtest[ytest.ravel()==1,0], xtest[ytest.ravel()==1,1], 'rx', label='test 2')
    legend(); colorbar();
    savefig(join(logdir, 'plot.jpg'))
    experiment.log_image(join(logdir, 'plot.jpg'))
    # show()


## make dataset
X, y = twospirals(args.ndata//2, noise=args.noise)
order = np.random.permutation(len(X))
X = X[order]
y = y[order]
splitIdx = int(.5*len(X))
xtrain, ytrain = X[:splitIdx], y[:splitIdx, None]
xtest, ytest = X[splitIdx:], y[splitIdx:, None]
if args.batchsize==None: args.batchsize = len(xtrain); print('fullbatch gradient descent')

# make model
model = Model(args)
model.fit(xtrain, ytrain, xtest, ytest)
model.plot(xtrain, ytrain, xtest, ytest)
print('done!')