import time
import numpy as np
import tensorflow as tf
import argparse
import matplotlib.pyplot as plt

tf.reset_default_graph()
seed = 1234
np.random.seed(seed)

def get_args():
  parser = argparse.ArgumentParser(description='linear network')
  parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
  parser.add_argument('--eta', default=0.1, type=float, help='eta from the gradient confusion condition (see paper)')
  parser.add_argument('--ndim', default=50, type=int, help='number of input dimensions')
  parser.add_argument('--nclass', default=1, type=int, help='number of output classes')
  parser.add_argument('--ndata', default=50, type=int, help='number of examples')
  parser.add_argument('--width', default=500, type=int, help='width: number of units in each hidden layer')
  parser.add_argument('--depth', default=0, type=int, help='depth: number of hidden layers (exclusive of input and output layers)')
  parser.add_argument('--activation', default='identity', help='activation function: can be "identity", "relu", or "tanh"')
  parser.add_argument('--nepoch', default=1, type=int, help='number of epochs')
  parser.add_argument('--batchsize', default=1, type=int, help='batchsize to use for training')
  parser.add_argument('--testbatchsize', default=1, type=int, help='batchsize to use for test')
  parser.add_argument('--train_frac', default=0, type=float, help='fraction of data used for training, rest for test')
  parser.add_argument('--savestr', default='debug.npz', help='file to save data')
  parser.add_argument('--mem_frac', default=.95, type=float, help='fraction of memory to use')

  # parse args
  args = parser.parse_args()
  return args

def corr_agg(corr, saveid=None):
  '''Aggregate statistics for the covariance matrix. Input: correlation matrix of shape n x n where n is number of points.
     Output: dictionary of aggregate statistics'''

  # get the norms
  norms = np.sqrt(corr[np.diag_indices(corr.shape[0],2)])
  normpair = np.dot(norms[:,None], norms[None,:])

  # get rid of diagonal elements
  corr[np.diag_indices(corr.shape[0],2)] = 0

  # get normalized correlation (cosine similarity)
  corr = corr/normpair

  # OPTIONAL plot hist of correlations
  if saveid!=None: plt.hist(corr.ravel(),100); plt.title('ndim='+str(saveid)); plt.xlim([-1,1]); plt.ylim([0, 3000]); plt.savefig(str(saveid/1000.)+'.jpg');

  prob_viol = np.sum(corr<-args.eta)/(corr.size-corr.shape[0])
  return dict(prob_viol=prob_viol, mea=corr.mean(), std=corr.std(), min=corr.min(), max=corr.max())

args = get_args()

## ----- INITIALIZATIONS -------

# precompute stuff
nunit = [args.width for i in range(args.depth)]
nunit = [args.ndim] + nunit + [args.nclass]
ntrain, ntest = int(args.train_frac * args.ndata), int((1 - args.train_frac) * args.ndata)
nbatchtrain = int(ntrain / args.batchsize)
nbatchtest = int(ntest / args.testbatchsize)

# generate data
xdata = np.random.randn(args.ndata, args.ndim)
xdata = xdata / np.linalg.norm(xdata, axis=1)[:, None]
ydata = np.mean(xdata, axis=1)[:, None] # feel free to redefine this labeling function
xtrain, xtest = xdata[:ntrain, :], xdata[ntrain:, :]
ytrain, ytest = ydata[:ntrain, :], ydata[ntrain:, :]

## ----- BUILD GRAPH -------

# inputs to network
inputs = tf.placeholder(dtype=tf.float32, shape=(None, args.ndim), name='inputs')
labels = tf.placeholder(dtype=tf.float32, shape=(None, args.nclass), name='labels')
weights = []

# activation function
if args.activation=='identity': activation_func = tf.identity
elif args.activation=='relu': activation_func = tf.nn.relu
elif args.activation=='tanh': activation_func = tf.tanh
else: print('UNKNOWN ACTIVATION FUNCTION! using identity'); activation_func = tf.identity

# forward prop
a = inputs
for l in range(1, len(nunit)):
  w = tf.get_variable(dtype=tf.float32, shape=(nunit[l - 1], nunit[l]), name='weights_' + str(l),
                      initializer=tf.random_normal_initializer(0,1/np.sqrt(nunit[l-1])))
  weights = weights + [w]
  a = tf.matmul(a, w, name='layer_' + str(l))
  a = activation_func(a)
loss = tf.nn.l2_loss(a - labels)

# gradient operations
optim = tf.train.GradientDescentOptimizer(learning_rate=args.lr)
gradients = tf.gradients(loss, weights)
train_op = optim.apply_gradients(zip(gradients, weights))
grad = tf.concat([tf.reshape(g, [-1]) for g in gradients], axis=0)

## ----- RUN SESSION -------
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.mem_frac)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
  sess.run(tf.global_variables_initializer())

  # loop over epochs
  train, test = [], []
  for epoch in range(args.nepoch):

    # # TRAIN
    # tic = time.time()
    # running_loss, running_grad = 0, np.empty((0, grad.get_shape().as_list()[0]))
    # for bi in range(nbatchtrain):
    #   _, lossdata, graddata = sess.run([train_op, loss, grad], feed_dict={inputs: xtrain[bi:bi + args.batchsize, :],
    #                                                                       labels: ytrain[bi:bi + args.batchsize, :]})
    #   running_loss += lossdata
    #   running_grad = np.append(running_grad, [graddata], axis=0)

    # cagg = corr_agg(corr)
    # gradnorm = np.linalg.norm(running_grad) / nbatchtrain
    # train = train + [dict(cagg=cagg, ave_loss=running_loss / ntrain, gradnorm=gradnorm)]
    # print('TEST\tepoch=' + str(epoch) + '\tloss=' + str(running_loss / ntest)
    #       + '\t\tprob_viol=' + str(cagg['prob_viol'])
    #       + '\t\ttime_elapsed=' + str(time.time() - tic))

    # TEST
    tic = time.time()
    running_loss, running_grad = 0, np.empty((0, grad.get_shape().as_list()[0]))
    for bi in range(nbatchtest):
      lossdata, graddata = sess.run([loss, grad], feed_dict={inputs: xtest[bi:bi + args.testbatchsize, :],
                                                             labels: ytest[bi:bi + args.testbatchsize, :]})
      running_loss += lossdata
      running_grad = np.append(running_grad, [graddata], axis=0)
    corr = np.dot(running_grad, running_grad.T)
    cagg = corr_agg(corr, None)
    gradnorm = np.linalg.norm(running_grad) / nbatchtest
    test = test + [dict(cagg=cagg, ave_loss=running_loss / ntest, gradnorm=gradnorm)]
    print('TEST\tepoch=' + str(epoch) + '\tloss=' + str(running_loss / ntest)
          + '\t\tprob_viol=' + str(cagg['prob_viol'])
          + '\t\ttime_elapsed=' + str(time.time() - tic))

np.savez(args.savestr, train=train, test=test)
time.sleep(.1)
