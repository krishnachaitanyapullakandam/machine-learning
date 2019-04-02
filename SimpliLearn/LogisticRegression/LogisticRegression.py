from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import metrics

digits = load_digits()

print ("Image Data Shape", digits.data.shape)
print ("Label Data Shape", digits.target.shape)

plt.figure(figsize=(20,4))
for index, (image, label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):
	plt.subplot(1,5,index+1)
	plt.imshow(np.reshape(image, (8,8)), cmap=plt.cm.gray)
	plt.title('Training: %i\n' % label, fontsize = 20)
plt.show()

# Dividing the dataset into Training and Test set
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.23, random_state=2)

print('Shape of X_Training set : ', x_train.shape)
print('Shape of Y_Training set : ', y_train.shape)
print('Shape of X_Testing set : ', x_test.shape)
print('Shape of Y_Testing set : ', y_test.shape)

# Making an instance of the model and training it
logisticRegr = LogisticRegression()
logisticRegr.fit(x_train, y_train)

# Predicting the output of the first element of the test set
print(logisticRegr.predict(x_test[0].reshape(1,-1)))

# Predicting the entire dataset
predictions = logisticRegr.predict(x_test)

# Determining the accuracy of the model
score = logisticRegr.score(x_test, y_test)
print(score)

cm = metrics.confusion_matrix(y_test, predictions)
print(cm)

#Representing the confusion matrix in a heat map
plt.figure(figsize=(9,9))
sb.heatmap(cm, annot=True, fmt=".3f", linewidth=0.5, square=True, cmap="Blues_r");
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy score: [0]'.format(score)
plt.title(all_sample_title, size=15);

index = 0
misclassifiedIndex = []
for predict, actual in zip(predictions, y_test):
	if predict==actual:
		misclassifiedIndex.append(index)
	index += 1

plt.figure(figsize=(20,3))
for plotIndex, wrong in enumerate(misclassifiedIndex[0:4]):
	plt.subplot(1,4,plotIndex+1)
	plt.imshow(np.reshape(x_test[wrong], (8,8)), cmap=plt.cm.gray)
	plt.title("Predicted: {}, Actual: {}".format(predictions[wrong], y_test[wrong]), fontsize=20)
plt.show()

index = 0
classifiedIndex = []
for predict, actual in zip(predictions, y_test):
	if predict==actual:
		classifiedIndex.append(index)
	index += 1

plt.figure(figsize=(20,3))
for plotIndex, wrong in enumerate(classifiedIndex[0:4]):
	plt.subplot(1,4,plotIndex+1)
	plt.imshow(np.reshape(x_test[wrong], (8,8)), cmap=plt.cm.gray)
	plt.title("Predicted: {}, Actual: {}".format(predictions[wrong], y_test[wrong]), fontsize=20)
plt.show()