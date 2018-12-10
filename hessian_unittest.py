##
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from hessian_unittest_utils import *
np.random.seed(322)

# build model and weight vector
ndim = 2
print('building graph')
model = Model(ndim, nminima=20)
print('done')

# start tf session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# get loss and biggest-eigenvalue surfaces
L, bigEig, w1, w2 = loss_surf(sess, model, b=1.3, npts=200, getBigEig=True)

## plot
def plot_cost(beta):

    # beta, cb = 3e-7, 3
    cb = 3
    bigEigSqr = bigEig**2
    cost = L+beta*bigEigSqr
    plt.figure(figsize=(8,6))
    ax0 = plt.subplot2grid((2, 3), (0, 0))
    im = ax0.imshow(L.T, extent=(w1.min(), w1.max(), w2.min(), w2.max()),
                    clim=[-1.5,1.5])
                    # clim=L.mean()+L.std()*cb*np.array([-1,1]))
    ax0.set_title('Loss')
    ax1 = plt.subplot2grid((2, 3), (1, 0))
    im = ax1.imshow(bigEigSqr.T, extent=(w1.min(), w1.max(), w2.min(), w2.max()),
                    clim=bigEig.mean()+bigEigSqr.std()*.05*np.array([-1,1]))
    ax1.set_title('Eig^2')
    ax2 = plt.subplot2grid((2, 3), (0, 1), colspan=2, rowspan=2)
    im = ax2.imshow(cost.T, extent=(w1.min(), w1.max(), w2.min(), w2.max()),
                    clim=[-1.5,1.5])
                    # clim=cost.mean()+cost.std()*cb*np.array([-1,1]))
    ax2.set_title('Loss + Beta*Eig^2 | Beta='+str(beta))
    plt.tight_layout()

# ##
# beta = 1e-7
# plot_cost(beta)
# plt.show()

betas = 8e-5*np.linspace(0,1,30)
for i, beta in enumerate(betas):
    plot_cost(beta)
    plt.savefig(str(i*1e-3)+'.jpg')
    if np.mod(i,4)==0: plt.show()
