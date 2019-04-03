import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets.samples_generator import make_blobs

# We create 40 separabale points
x,y = make_blobs(n_samples=40,centers=2,random_state=20)
clf = svm.SVC(kernel='linear', C=1)
clf.fit(x,y)

# Display the data in graph form
plt.scatter(x[:,0],x[:,1],c=y,s=30,cmap=plt.cm.Paired)
# plt.show()

# Using to predict unknonw data
newData = [[3,4],[5,6]]
print(clf.predict(newData))


# Fit the model, don't regularize for illustration purposes
clf = svm.SVC(kernel='linear', C=1000)
clf.fit(x, y)

plt.scatter(x[:,0],x[:,1],c=y,s=30,cmap=plt.cm.Paired)
# plt.show

# plot the decision function
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30) 
yy, xx = np.meshgrid(yy, xx)
xy = np.vstack([xx.ravel(), yy.ravel()]).T
z = clf.decision_function(xy).reshape(xx.shape)

# plot decision boundary and margins
ax.contour(xx, yy, z, colors='k', levels=[-1,0,1], alpha=0.5, linestyle=['--','-','--'])
ax.scatter(clf.support_vectors_[:,0], clf.support_vectors_[:,1], s=100, linewidth=1, facecolors='none')
plt.show()