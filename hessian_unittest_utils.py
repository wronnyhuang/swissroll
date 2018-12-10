import tensorflow as tf
import numpy as np


class Model(object):
  '''model for calculating a simple loss as a function of weights. Also calculates hessian vector product'''

  def __init__(self, ndim, nminima=1):
    '''build a loss surface and compute the hessian vector product. ndim is the dimensionality of the weights'''

    # calculate simple loss
    self.ndim = ndim
    self.weights = tf.placeholder(name='weights', shape=[None, self.ndim], dtype=tf.float32)
    self.loss = self.gen_loss_fun(nminima)
    self.gradLoss = tf.gradients(self.loss, self.weights)[0]

    # calculate hessian vector product
    self.randUnitVec = tf.placeholder(name='randUnitVec', dtype=tf.float32, shape=[None, self.ndim])
    self.gradVecProd = tf.reduce_sum(tf.multiply(self.gradLoss, self.randUnitVec), axis=1)
    self.hessVecProd = tf.gradients(self.gradVecProd, self.weights)[0]
    # self.hessVecProd = tf.gradients(self.gradLoss, self.weights, self.randUnitVec)[0]
    self.normHv = tf.norm(self.hessVecProd, axis=1)
    self.unitHv = tf.transpose(tf.divide(tf.transpose(self.hessVecProd), self.normHv))

  def gen_single_minima(self, coeff, mu, amplitude):
    '''unit function for gen_loss_fun(). returns a loss function with a single minima'''
    return tf.multiply(amplitude, tf.negative(
      tf.exp(tf.negative(tf.reduce_sum(tf.pow(tf.multiply(tf.subtract(self.weights, mu), coeff), 2), axis=1)))))

  def gen_loss_fun(self, nminima=1):
    '''generate a random loss function'''

    # generate the first minima
    width = tf.constant(.8, dtype=tf.float32)
    coeff = tf.reciprocal(width)
    centroid = tf.constant([0, 0], dtype=tf.float32)
    amplitude = tf.constant(1, dtype=tf.float32)
    losses = [self.gen_single_minima(coeff=coeff, mu=centroid, amplitude=amplitude)]

    # generate remaining minima randomly
    for i in range(nminima - 1):
      width = tf.constant(np.random.randn(1) * 0.14, dtype=tf.float32)
      coeff = tf.reciprocal(width)
      centroid = tf.constant((np.random.rand(self.ndim) - .5) * 2.2, dtype=tf.float32)
      amplitude = tf.constant(np.random.randn(1) * .2 + 1, dtype=tf.float32)
      losses.append(self.gen_single_minima(coeff, centroid, amplitude))
    loss = tf.add_n(losses)
    return loss

  # get largest eigenvalue/vector of the hessian
  def get_largest_eig(self, sess, w, rv=None):
    '''get the largest eigenvalue/vector'''
    if rv is None: rv = rand_unit_vec(w.shape)
    eigVec = rv
    for i in range(30):
      eigVec, eigVal = sess.run([self.unitHv, self.normHv], feed_dict={self.randUnitVec: eigVec, self.weights: w})
      print('iteration ' + str(i) + ', eig=' + str(eigVal[5000]))
    return eigVec, eigVal

  # update random vector by multiplying with hessian
  # todo this is not ready for use. does not support batches
  def get_all_eig(self, sess, w):
    '''use power method to obtain all eigenvalues and eigenvectors'''

    rv = rand_unit_vec(w.shape)
    for dim in range(self.ndim):
      rv = rv / np.linalg.norm(rv, axis=1)[:, None]
      eigVec, eigVal = self.get_largest_eig(sess, w, rv)
      if dim == 0:
        eigVals, eigVecs = eigVal, eigVec
      else:
        eigVals = np.append(eigVals, eigVal)
        eigVecs = np.vstack((eigVecs, eigVec))
      rv = rv - rv.dot(eigVec) * eigVec
      rv[np.abs(rv) < 1e-6] = 0.

    return eigVecs, eigVals


def rand_unit_vec(shape):
  '''return a random unit vector'''
  rv = np.random.rand(*shape) * 2 - 1
  normrv = np.linalg.norm(rv, axis=1)
  return rv / normrv[:, None]


def loss_surf(sess, model, b=2, npts=30, getBigEig=True):
  '''compute the loss surface and biggest-eigenvalue surface'''

  # setup meshgrid
  w1, w2 = np.linspace(-b, b, npts), np.linspace(-b, b, npts)
  W2, W1 = np.meshgrid(w1, w2)
  W = np.stack((W1.ravel(), W2.ravel()), axis=1)
  L = sess.run(model.loss, feed_dict={model.weights: W})
  if getBigEig: bigEigVec, bigEig = model.get_largest_eig(sess, W)

  # rehape results
  L = L.reshape(W1.shape)
  bigEig = bigEig.reshape(W1.shape)
  return L, bigEig, w1, w2
