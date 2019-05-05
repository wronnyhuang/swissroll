import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

with open('./poison-opt/wiggle_data.pickle', 'rb') as f:
    data = pickle.load(f)

print(data['X'].shape)
print(data['Y'].shape)
print(data['xent'])
X = np.array([data['X'][i][0] for i in range(data['X'].shape[0])])
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(X, X, data['xent'], rcount=500, ccount=500)
plt.show()
