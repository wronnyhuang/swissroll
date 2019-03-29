from comet_ml import Experiment
import pickle
import numpy as np
import cv2
from matplotlib.pyplot import plot, imshow, colorbar, show, axis, hist, subplot, xlabel, ylabel, title, legend, savefig, figure, close, suptitle, tight_layout, contourf, xlim, ylim
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys
import random
from os.path import join, basename, dirname, exists
from glob import glob
import tensorflow as tf
import argparse
import utils
from time import time, sleep

parser = argparse.ArgumentParser(description='model')
parser.add_argument('-gpu', default=0, type=int)
parser.add_argument('-sugg', default='debug', type=str)
parser.add_argument('-noise', default=1, type=float)
parser.add_argument('-tag', default='', type=str)
# lr and schedule
parser.add_argument('-lr', default=.0067, type=float)
parser.add_argument('-lrlog', default=-2, type=float)
parser.add_argument('-lrstep', default=600, type=int)
parser.add_argument('-lrstep2', default=3000, type=int)
parser.add_argument('-lrstep3', default=1e9, type=int)
parser.add_argument('-nepoch', default=10000, type=int)
# poisoning
parser.add_argument('-perfect', action='store_true')
parser.add_argument('-distrfrac', default=0, type=float)
parser.add_argument('-distrstep', default=8812, type=int)
parser.add_argument('-distrstep2', default=18142, type=int)
# regularizers
parser.add_argument('-wdeccoef', default=0, type=float)
parser.add_argument('-speccoef', default=0, type=float)
parser.add_argument('-speccoeflog', default=-2, type=float)
parser.add_argument('-projvec_beta', default=0, type=float)
parser.add_argument('-warmupStart', default=800, type=int)
parser.add_argument('-warmupPeriod', default=1000, type=int)
# sharpfinder
parser.add_argument('-randvec', action='store_true')
parser.add_argument('-ngradspec', default=300, type=int)
# saving and restorin
parser.add_argument('-save', action='store_true')
parser.add_argument('-pretrain_dir', default=None, type=str)
# hidden hps
parser.add_argument('-nhidden', default=[10,10,10,10,10,10], type=int, nargs='+')
parser.add_argument('-nhidden1', default=8, type=int)
parser.add_argument('-nhidden2', default=14, type=int)
parser.add_argument('-nhidden3', default=20, type=int)
parser.add_argument('-nhidden4', default=26, type=int)
parser.add_argument('-nhidden5', default=32, type=int)
parser.add_argument('-nhidden6', default=32, type=int)
# experiment hps
parser.add_argument('-batchsize', default=None, type=int)
parser.add_argument('-ndim', default=2, type=int)
parser.add_argument('-nclass', default=1, type=int)
parser.add_argument('-ndata', default=400, type=int)
parser.add_argument('-max_grad_norm', default=10, type=float)
parser.add_argument('-seed', default=1234, type=int)
# wiggle
parser.add_argument('-wiggle', action='store_true')
parser.add_argument('-saveplotdata', action='store_true')
parser.add_argument('-span', default=.5, type=float)
parser.add_argument('-nspan', default=100, type=int)
parser.add_argument('-along', default='random', type=str)
args = parser.parse_args()

def twospirals(n_points, noise=.5):
    """
     Returns the two spirals dataset.
    """
    n = np.sqrt(np.random.rand(n_points,1)) * 780 * (2*np.pi)/360
    d1x = -1.5*np.cos(n)*n + np.random.randn(n_points,1) * noise
    d1y =  1.5*np.sin(n)*n + np.random.randn(n_points,1) * noise
    return (np.vstack((np.hstack((d1x,d1y)),np.hstack((-d1x,-d1y)))),
            np.hstack((np.zeros(n_points),np.ones(n_points))))

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

    # weight decay
    regularizable = [t for t in tf.trainable_variables() if t.op.name.find('bias')==-1]
    wdec = tf.global_norm(regularizable)**2
    self.loss = self.xent + self.args.wdeccoef*wdec # + self.speccoef*self.spec

    # gradient operations
    optim = tf.train.AdamOptimizer(self.lr)
    grads = tf.gradients(self.loss, tf.trainable_variables())

    if args.speccoef != 0:
      specs = []
      self.projvec_op = []
      for i in range(args.ngradspec):
        print('creating curvature graph '+str(i))
        spec, projvecop, self.projvec_corr, self.eigvec, self.norm_values = spectral_radius(self.xent, tf.trainable_variables(), self.args.projvec_beta, i)
        specs.append(spec)
        self.projvec_op.append(projvecop)
      self.spec = tf.add_n(specs) / args.ngradspec
      gradspec = tf.gradients(self.spec, tf.trainable_variables())
      gradspec, self.gradspecnorm = tf.clip_by_global_norm(gradspec, clip_norm=self.args.max_grad_norm)
      grads = [g1 + self.speccoef * g2 for g1,g2 in zip(grads, gradspec)]
    else:
      self.spec = self.gradspecnorm = self.gradspec = self.norm_values = self.eigvec = self.projvec_op = self.projvec_corr = tf.constant(False)

    self.gradnorm = tf.global_norm(grads)
    self.train_op = optim.apply_gradients(zip(grads, tf.trainable_variables()))

    # accuracy
    self.predictions = tf.sigmoid(logits)
    equal = tf.equal(self.labels, tf.round(self.predictions))
    self.acc = tf.reduce_mean(tf.to_float(equal))

  def setupTF(self):
    '''setup the tf session and load pretrained model if desired'''

    self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
    self.sess.run(tf.global_variables_initializer())

    # load pretrained model
    if args.pretrain_dir != None:
      utils.download_pretrained(logdir, pretrain_dir=args.pretrain_dir) # download it and put in logdir
      ckpt_file = join(logdir, 'model.ckpt')
      print('Loading pretrained model from '+ckpt_file)
      # var_list = list(set(tf.global_variables())-set(tf.global_variables('accum'))-set(tf.global_variables('projvec')))
      var_list = tf.trainable_variables()
      saver = tf.train.Saver(var_list=var_list, max_to_keep=1)
      saver.restore(self.sess, ckpt_file)

  def fit(self, xtrain, ytrain, xtest, ytest):
    '''fit the model to the data'''

    nbatch = len(xtrain)//self.args.batchsize

    # loop over epochs
    for epoch in range(self.args.nepoch):
      order = np.random.permutation(len(xtrain))
      xtrain = xtrain[order]
      ytrain = ytrain[order]

      # sample distribution
      xdistr, ydistr = twospirals(args.ndata//4, args.noise)
      ydistr = ydistr[:, None]
      if not args.perfect: ydistr = 1 - ydistr # flip the labels

      # antilearner schedule
      if epoch<self.args.distrstep: distrfrac = self.args.distrfrac
      elif epoch<self.args.distrstep2: distrfrac = self.args.distrfrac*2
      else: distrfrac = self.args.distrfrac*4

      # lr schedule
      if epoch<self.args.lrstep: lr = self.args.lr
      elif epoch<self.args.lrstep2: lr = self.args.lr/10
      elif epoch<self.args.lrstep3: lr = self.args.lr/100
      else: lr = self.args.lr/1000

      # speccoef schedule
      speccoef = self.args.speccoef*max(0, min(1, ( max(0, epoch - self.args.warmupStart) / self.args.warmupPeriod )**2 ))

      # loop over batches (usually the batchsize is just the dataset size) so there's only one iteration
      for b in self.args.batchsize * np.arange(nbatch):

        xbatch = np.concatenate([ xtrain[b:b + self.args.batchsize, :], xdistr[b:b + int(self.args.batchsize*distrfrac), :]])
        ybatch = np.concatenate([ ytrain[b:b + self.args.batchsize, :], ydistr[b:b + int(self.args.batchsize*distrfrac), :]])

        metrics = dict(xent=self.xent, acc=self.acc, spec=self.spec, )
        ops = [self.train_op, self.projvec_op]
        _, metrics, = self.sess.run([ops, metrics],
                                                  {self.inputs: xbatch,
                                                   self.labels: ybatch,
                                                   self.is_training: True,
                                                   self.lr: lr,
                                                   self.speccoef: speccoef,
                                                   })
        _, spec, = self.sess.run([self.projvec_op, self.spec],
                                    {self.inputs: xbatch,
                                     self.labels: ybatch,
                                     })

      if np.mod(epoch, 2)==0:

        experiment.log_metrics(metrics, step=epoch)
        print('TRAIN\tepoch=' + str(epoch) + '\txent=' + str(metrics['xent']) + '\tacc=' + str(metrics['acc']) +
              '\tspectrain = ' + str(metrics['spec']) + '\tspec = ' + str(spec))

        xent_test, acc_test = self.evaluate(xtest, ytest)
        experiment.log_metric('test/xent', xent_test, epoch)
        experiment.log_metric('test/acc', acc_test, epoch)
        print('TEST\tepoch=' + str(epoch) + '\txent=' + str(xent_test) + '\tacc=' + str(acc_test))

        experiment.log_metric('lr', lr, epoch)
        experiment.log_metric('speccoef', speccoef, epoch)

        # if args.speccoef == 0:
        #   # run several power iterations to get accurate hessian
        #   spec, _, projvec_corr, acc_clean, xent_clean = self.get_hessian(xtrain, ytrain)
        #   print('spec', spec, '\tprojvec_corr', projvec_corr)

  def get_hessian(self, xdata, ydata):
    '''return hessian info namely eigval, eigvec, and projvec_corr given the set of data'''
    for i in range(10):
      acc_clean, xent_clean, spec, _, projvec_corr, eigvec = \
        self.sess.run([self.acc, self.xent, self.spec, self.projvec_op, self.projvec_corr, self.eigvec],
          {self.inputs: xdata, self.labels: ydata, self.speccoef: args.speccoef})
    return spec, eigvec, projvec_corr, acc_clean, xent_clean

  def evaluate(self, xtest, ytest):
    '''evaluate input data (labels included) and get xent and acc for that dataset'''
    xent, acc = self.sess.run([self.xent, self.acc], {self.inputs: xtest, self.labels: ytest, self.is_training: False})
    return xent, acc

  def predict(self, xinfer):
    return self.sess.run([self.predictions], {self.inputs: xinfer, self.is_training: False})

  def infer(self, xinfer):
    '''inference on a batch of input data xinfer. outputs collapsed to 1 or 0'''
    yinfer = self.predict(xinfer)
    yinfer = yinfer[0]
    yinfer[yinfer > .5] = 1
    yinfer[yinfer <= .5] = 0
    return yinfer

  def plot(self, xtrain, ytrain, xtest=None, ytest=None, name='plot.jpg', plttitle='plot', index=0):
    '''plot decision boundary alongside loss surface'''

    # make contour of decision boundary
    xlin = 25*np.linspace(-1,1,300)
    xx1, xx2 = np.meshgrid(xlin, xlin)
    xinfer = np.column_stack([xx1.ravel(), xx2.ravel()])
    yinfer = self.infer(xinfer)
    yy = np.reshape(yinfer, xx1.shape)

    # plot the decision boundary
    figure(figsize=(8,6))
    plt.subplot2grid((3,4), (0,1), colspan=3, rowspan=3)
    contourf(xx1, xx2, yy, alpha=.3, cmap='rainbow')

    # plot blue class
    xinfer = xtrain[ytrain.ravel()==0]
    yinfer = self.infer(xinfer)
    plot( xinfer[yinfer.ravel()==0, 0], xinfer[yinfer.ravel()==0, 1], 'b.', markersize=8, label='class 1 correct' )
    plot( xinfer[yinfer.ravel()==1, 0], xinfer[yinfer.ravel()==1, 1], 'b.', markersize=4, label='class 1 error' )
    xinferblue, yinferblue = xinfer, yinfer

    # plot red class
    xinfer = xtrain[ytrain.ravel()==1]
    yinfer = self.infer(xinfer)
    plot( xinfer[yinfer.ravel()==1, 0], xinfer[yinfer.ravel()==1, 1], 'r.', markersize=8, label='class 1 correct' )
    plot( xinfer[yinfer.ravel()==0, 0], xinfer[yinfer.ravel()==0, 1], 'r.', markersize=4, label='class 1 error' )
    xinferred, yinferred = xinfer, yinfer

    axis('image'); title(plttitle); legend(loc='upper left', framealpha=.4); axis('off')

    # load data from surface plots
    if exists(join(logdir,'surface.pkl')):

      with open(join(logdir,'surface.pkl'), 'rb') as f:
        cfeed, xent, acc, spec = pickle.load(f)

      # surface of xent
      plt.subplot2grid((3,4), (0,0))
      plot(cfeed, xent, '-', color='orange')
      plot(cfeed[index], xent[index], 'ko', markersize=8)
      title('xent'); ylim(0, 5)
      plt.gca().axes.get_xaxis().set_ticklabels([])

      # surface of acc
      plt.subplot2grid((3,4), (1,0))
      plot(cfeed, acc, '-', color='green')
      plot(cfeed[index], acc[index], 'ko', markersize=8)
      title('acc'); ylim(0, 1.05)
      plt.gca().axes.get_xaxis().set_ticklabels([])

      # surface of spec
      plt.subplot2grid((3,4), (2,0))
      plot(cfeed, spec, '-', color='cyan')
      plot(cfeed[index], spec[index], 'ko', markersize=8)
      title('curv'); ylim(0, 12000)

    suptitle(args.sugg); tight_layout()

    # save the data needed to reproduce plot
    if args.saveplotdata:
      os.makedirs(join(logdir, 'plotdata'), exist_ok=True)
      with open(join(logdir, 'plotdata', name[:-4]+'.pkl'), 'wb') as f:
        pickle.dump(dict(cfeed=cfeed, xent=xent, acc=acc, spec=spec, xinferred=xinferred, yinferred=yinferred,
                         xinferblue=xinferblue, yinferblue=yinferblue, xx1=xx1, xx2=xx2, yy=yy, xtrain=xtrain, ytrain=ytrain), f)

    # image metadata and save image
    os.makedirs(join(logdir, 'images'), exist_ok=True)
    savefig(join(logdir, 'images', name))
    if name=='plot.jpg': experiment.log_image(join(logdir, 'images/plot.jpg')); os.remove(join(logdir, 'images/plot.jpg'))
    sleep(.1)
    close('all')

  def wiggle(self, xdata, ydata, span=1, along='random'):
    '''perturb weights and plot the decision boundary at each step, also get loss surface'''

    # produce random direction
    if along=='random':
      direction = utils.get_random_dir(self.sess)
      direction[-2] = direction[-2][:, None] # a hack to make it work
    elif along=='eigvec':
      eigval, direction, _, _, _ = self.get_hessian(xdata, ydata)

    # name of the surface sweep for comet
    name = 'span_' + str(args.span) + '/' + basename(args.pretrain_dir) + '/' + along # name of experiment

    # linspace of span
    cfeed = span/2 * np.linspace(-1, 1, args.nspan)

    # loop over all points along surface direction
    xent = np.zeros(len(cfeed))
    acc = np.zeros(len(cfeed))
    spec = np.zeros(len(cfeed))
    weights = self.sess.run(tf.trainable_variables())
    for i, c in enumerate(cfeed):

      # perturbe the weights
      perturbedWeights = [w + c * d for w, d in zip(weights, direction)]

      # visualize what happens to decision boundary when weights are wiggled
      self.assign_weights(perturbedWeights)
      if exists(join(logdir,'surface.pkl')): self.plot(xdata, ydata, name=str(i/1000)+'.jpg', plttitle=format(c,'.3f'), index=i)

      # compute the loss surface
      # xent[i], acc[i] = self.evaluate(xdata, ydata)
      spec[i], _, projvec_corr, acc[i], xent[i] = self.get_hessian(xtrain, ytrain)
      experiment.log_metric('xent', xent[i], step=i)
      experiment.log_metric('acc', acc[i], step=i)
      experiment.log_metric('spec', spec[i], step=i)

      print('progress:', i + 1, 'of', len(cfeed), '| xent:', xent[i])

    # make gif or save surface
    if exists(join(logdir,'surface.pkl')): # make gif of the decision boundary plots
      os.system('python make_gif.py '+join(logdir, 'images')+' '+join(logdir, 'wiggle.gif'))
      experiment.log_asset(join(logdir, 'wiggle.gif'))
      os.system('dbx upload '+join(logdir,'wiggle.gif')+' ckpt/swissroll/'+args.sugg+'/')

    # save the surface data
    with open(join(logdir, 'surface.pkl'), 'wb') as f:
      pickle.dump((cfeed, xent, acc, spec), f)
      experiment.log_asset(join(logdir, 'surface.pkl'))

  def assign_weights(self, weights):
    self.sess.run([tf.assign(t,w) for t,w in zip(tf.trainable_variables(), weights)])

  def save(self):
    '''save model'''
    ckpt_state = tf.train.get_checkpoint_state(logdir)
    ckpt_file = join(logdir, 'model.ckpt')
    print('Saving model to '+ckpt_file)
    saver = tf.train.Saver(max_to_keep=1)
    saver.save(self.sess, ckpt_file)
    os.system('dbx upload '+logdir+' ckpt/swissroll/')

def spectral_radius(xent, regularizable, projvec_beta=.55, iter=0):
  """returns principal eig of the hessian"""

  # create initial projection vector (randomly and normalized)
  projvec_init = [np.random.randn(*r.get_shape().as_list()) for r in regularizable]
  magnitude = np.sqrt(np.sum([np.sum(p**2) for p in projvec_init]))
  projvec_init = projvec_init/magnitude

  # projection vector tensor variable
  with tf.variable_scope('projvec'+str(iter)):
    projvec = [tf.get_variable(name=r.op.name, dtype=tf.float32, shape=r.get_shape(),
                               trainable=False, initializer=tf.constant_initializer(p))
               for r,p in zip(regularizable, projvec_init)]

  # layer norm
  # norm_values = utils.layernormdev(regularizable)
  norm_values = utils.filtnorm(regularizable)
  # norm_values = [tf.ones_like(t) for t in tf.trainable_variables()]
  projvec_mul_normvalues = [tf.multiply(f,p) for f,p in zip(norm_values, projvec)]

  # get the hessian-vector product
  gradLoss = tf.gradients(xent, regularizable)
  hessVecProd = tf.gradients(gradLoss, regularizable, projvec_mul_normvalues)
  hessVecProd = [h*n for h,n in zip(hessVecProd, norm_values)]

  # principal eigenvalue: project hessian-vector product with that same vector
  xHx = utils.list2dotprod(projvec, hessVecProd)

  # comopute next projvec
  if args.randvec:
    nextProjvec = [tf.random_normal(shape=p.get_shape()) for p in projvec]
  else:
    normHv = utils.list2norm(hessVecProd)
    unitHv = [tf.divide(h, normHv) for h in hessVecProd]
    nextProjvec = [tf.add(h, tf.multiply(p, projvec_beta)) for h,p in zip(unitHv, projvec)]
  normNextPv = utils.list2norm(nextProjvec)
  nextProjvec = [tf.divide(p, normNextPv) for p in nextProjvec]

  # diagnostics: dotprod and euclidean distance of new projection vector from previous
  projvec_corr = utils.list2dotprod(nextProjvec, projvec)

  # op to assign the new projection vector for next iteration
  with tf.control_dependencies([projvec_corr]):
    with tf.variable_scope('projvec_op'):
      projvec_op = [tf.assign(p,n) for p,n in zip(projvec, nextProjvec)]

  return xHx, projvec_op, projvec_corr, projvec_mul_normvalues, norm_values

if __name__ == '__main__':

  args.speccoef = 10**args.speccoeflog
  args.lr = 10**args.lrlog
  experiment = Experiment(api_key="vPCPPZrcrUBitgoQkvzxdsh9k", parse_args=False,
                          project_name='swissroll'+args.tag, workspace="wronnyhuang")
  home = os.environ['HOME']
  tf.reset_default_graph()
  logdir = '/root/ckpt/swissroll/'+args.sugg
  os.makedirs(logdir, exist_ok=True)
  open(join(logdir,'comet_expt_key.txt'), 'w+').write(experiment.get_key())
  if any([a.find('nhidden1')!=-1 for a in sys.argv[1:]]):
    args.nhidden = [args.nhidden1, args.nhidden2, args.nhidden3, args.nhidden4, args.nhidden5, args.nhidden6]
  experiment.log_parameters(vars(args))
  experiment.set_name(args.sugg)
  print(' '.join(sys.argv))

  os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
  np.random.seed(args.seed)
  tf.set_random_seed(args.seed)

  # make dataset
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
  if args.wiggle:
    model.wiggle(xtrain, ytrain, args.span, args.along)
  else:
    model.fit(xtrain, ytrain, xtest, ytest)
    model.plot(xtrain, ytrain)
    if args.save: model.save()
  print('done!')