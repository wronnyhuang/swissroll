import numpy as np
import cv2
from matplotlib.pyplot import plot, imshow, colorbar, show, axis, legend, contourf
from PIL import Image
import os
import random
from os.path import join, basename, dirname
from glob import glob
import tensorflow as tf
import argparse

home = os.environ['HOME']
tf.reset_default_graph()

parser = argparse.ArgumentParser(description='model')
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--nepoch', default=4000, type=int)
parser.add_argument('--ndim', default=2, type=int)
parser.add_argument('--nclass', default=1, type=int)
parser.add_argument('--nhidden', default=[6, 6, 7, 8], type=int, nargs='+')
parser.add_argument('--ndata', default=300, type=int)
parser.add_argument('--batchsize', default=40, type=int)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)





def twospirals(n_points, noise=.5):
    """
     Returns the two spirals dataset.
    """
    n = np.sqrt(np.random.rand(n_points,1)) * 650 * (2*np.pi)/360
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


    # forward prop
    a = self.inputs
    a = tf.layers.batch_normalization(a, training=self.is_training)

    # # with batchnorm
    # for l, nunit in enumerate( self.args.nhidden + [self.args.nclass] ):
    #   a = tf.layers.batch_normalization(a, training=self.is_training)
    #   a = tf.nn.relu(a)
    #   a = tf.layers.dense(a, nunit, use_bias=False)
    # logits = tf.layers.batch_normalization(a, training=self.is_training)

    # withtout batchnorm
    for l, nunit in enumerate( self.args.nhidden ):
      a = tf.layers.dense(a, nunit, use_bias=True, activation='relu')
    logits = tf.layers.dense(a, self.args.nclass)

    # losses
    xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=logits)
    self.xent = tf.reduce_mean(xent)

    # accuracy
    self.predictions = tf.sigmoid(logits)
    equal = tf.equal(self.labels, tf.round(self.predictions))
    self.acc = tf.reduce_mean(tf.to_float(equal))

    # gradient operations
    optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
    # # optim = tf.train.AdamOptimizer(self.lr)
    # grads = tf.gradients(self.xent, tf.trainable_variables())
    # self.train_op = optim.apply_gradients(zip(grads, tf.trainable_variables()))
    self.train_op = optim.minimize(self.xent)


  def setupTF(self):
    self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
    self.sess.run(tf.global_variables_initializer())

  def fit(self, xtrain, ytrain, xtest, ytest):
    nbatch = self.args.ndata//self.args.batchsize

    # loop over epochs
    for epoch in range(self.args.nepoch):
      order = np.random.permutation(len(xtrain))
      xtrain = xtrain[order]
      ytrain = ytrain[order]
      for bi in range(nbatch):
        if epoch<1000: lr = self.args.lr
        else: lr = self.args.lr/3
        # elif epoch<3000: lr = self.args.lr/10
        _, xent = self.sess.run([self.train_op, self.xent], {self.inputs: xtrain[bi:bi + args.batchsize, :],
                                                             self.labels: ytrain[bi:bi + args.batchsize, :],
                                                             self.is_training: True,
                                                             self.lr: lr})
      if np.mod(epoch, 100)==0:
        xent, acc = self.sess.run([self.xent, self.acc], {self.inputs:xtrain, self.labels:ytrain, self.is_training: False})
        print('TRAIN\tepoch=' + str(epoch) + '\txent=' + str(xent) + '\tacc=' + str(acc))
        xent, acc = self.sess.run([self.xent, self.acc], {self.inputs:xtest, self.labels:ytest, self.is_training: False})
        print('TEST\tepoch=' + str(epoch) + '\txent=' + str(xent) + '\tacc=' + str(acc))
        self.plot(xtrain, ytrain, xtest, ytest)


  def evaluate(self, xtest, ytest):
    acc = self.sess.run([self.acc], {self.inputs: xtest, self.labels: ytest, self.is_training: False})
    print('TEST\tacc='+str(acc))
    return acc

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

    plot(xtrain[ytrain.ravel()==0,0], xtrain[ytrain.ravel()==0,1], 'b.', label='class 1')
    plot(xtrain[ytrain.ravel()==1,0], xtrain[ytrain.ravel()==1,1], 'r.', label='class 2')
    plot(xtest[ytest.ravel()==0,0], xtest[ytest.ravel()==0,1], 'bx', label='class 1')
    plot(xtest[ytest.ravel()==1,0], xtest[ytest.ravel()==1,1], 'rx', label='class 2')
    legend()
    show()



## make dataset
X, y = twospirals(args.ndata//2, noise=.2)
order = np.random.permutation(len(X))
X = X[order]
y = y[order]
splitIdx = int(.9*len(X))
xtrain, ytrain = X[:splitIdx], y[:splitIdx, None]
xtest, ytest = X[splitIdx:], y[splitIdx:, None]

# make model
model = Model(args)
model.fit(xtrain, ytrain, xtest, ytest)
# model.evaluate(xtrain, ytrain)
# model.evaluate(xtest, ytest)



