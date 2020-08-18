import numpy as np
from sklearn.datasets import load_iris

#%%
iris = load_iris()
print('Iris Data key: {}'.format(iris.keys()))
print('Iris Data shape: {}'.format(iris.data.shape))
print('Iris Data target name: {}'.format(iris.target_names))
print('iris Data target: {}'.format(iris.target))
print('iris Data feature name: {}'.format(iris.feature_names))

#%%
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = \
    train_test_split(iris.data, iris.target, test_size=0.3, random_state=0)

#%%
# k-Nearest Neighbors Algorithm Steps.
# 1. Choose the number of k and a distance metric.
# 2. Find the k-nearest neighbors of the data record that we want to classify.
# 3. Assign the class label by majority vote.
def kNearestNeighbors(k, trainingData, trainingDataLabel, testData):
    # (numpy.ndarray.shape)
    #   Tuple of array dimensions.
    numTestData = np.shape(testData)[0]
    print('Number of Test Data:', numTestData)
    
    # (numpy.zeros)
    #   Return a new array of given shape and type, filled with zeros.
    predictData = np.zeros(numTestData)
    print('Predict Data:', predictData)

    for n in range(numTestData):
        # Euclidean Distance
        euclideanDistance = np.sum((trainingData - testData[n,:]) ** 2, axis=1)

        # Identify the nearest neighbors
        # (numpy.argsort)
        #   Returns the indices that would sort an array.
        index = np.argsort(euclideanDistance, axis=0)
        # (numpy.unique)
        #   Find the unique elements of an array.
        label = np.unique(trainingDataLabel[index[:k]])
        
        print('Distance:', euclideanDistance)
        print('Indices:', index)
        print('Labels:', label)

        if len(label) == 1:
            predictData[n] = np.unique(label)
            print('Predict', predictData[n])
        else:
            counts = np.zeros(max(label)+1)
            for i in range(k):
                counts[trainingDataLabel[index[i]]] += 1
            predictData[n] = np.max(counts)
            print(predictData[n])
        
    return predictData

#%%
from sklearn import preprocessing

# (LabelEncoder)
#   Encode target labels with value between 0 and n_classes-1.
labelEncoder = preprocessing.LabelEncoder()

irisTrainingData = X_train[1:,]
irisTrainingDataLabel = labelEncoder.fit_transform(Y_train[1:,])

predict = kNearestNeighbors(3, irisTrainingData, irisTrainingDataLabel, X_test)
print('Predict:', predict)

# (numpy.ndarray.astype)
#   Copy of the array, cast to a specified type.
predictRev = labelEncoder.inverse_transform(predict.astype(int))
print('Predict Revision:', predictRev)

#%%
# Classification Accuracy.
from sklearn import metrics

print('Classification Accuracy:', metrics.accuracy_score(Y_test, predict))
