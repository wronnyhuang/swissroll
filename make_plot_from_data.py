import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

with open('./wiggle_data_clean.pickle', 'rb') as f:
    data = pickle.load(f)

#with open('./poison-opt/wiggle_data.pickle', 'rb') as f:
#    data = pickle.load(f)

X = data['X']
Y = data['Y']
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, np.log10(data['xent']), rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
plt.savefig('xent_clean')
plt.show()

with open('./wiggle_data_poison.pickle', 'rb') as f:
    data = pickle.load(f)

#with open('./poison-opt/wiggle_data.pickle', 'rb') as f:
#    data = pickle.load(f)

X = data['X']
Y = data['Y']
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, np.log10(data['xent']), rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
plt.savefig('xent_poison')
plt.show()
