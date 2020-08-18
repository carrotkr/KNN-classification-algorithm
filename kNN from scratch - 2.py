import pandas as pd
import numpy as np

#%%
dataDir = '/Users/carrotkr/Downloads/'

#%%
# Feature Engineering
def prepareTitanicData(dataFrame):
    # We'll keep only these columns.
    cols = ['Pclass','Sex','Age','SibSp']
    
    X = dataFrame[cols]
    
    # Now do one-hot encoding for categorical variables.
    oneHot = pd.get_dummies(X['Pclass'])
    X = X.drop(['Pclass'], axis = 1)
    X = X.join(oneHot)
    
    oneHot = pd.get_dummies(X['Sex'])
    X = X.drop(['Sex'], axis = 1)
    X['female'] = oneHot['female']
    
    # Impute missing age values.
    X['Age'] = X['Age'].fillna(X['Age'].mean())
    
    # Standardize the data.
    X = (X - X.mean())/X.std()
    return X

#%%
# Train Data.
dataFrame_train = pd.read_csv(dataDir + 'titanic_train.csv')

Y_train = dataFrame_train['Survived']
Y_trainArray = Y_train.values

X_train = prepareTitanicData(dataFrame_train)
X_trainArray = X_train.values

#%%
# Test Data.
dataFrame_test = pd.read_csv(dataDir + 'titanic_test.csv')

X_test = prepareTitanicData(dataFrame_test)
X_testArray = X_test.values

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

titanicTrainingData = X_trainArray[1:,]
titanicTrainingDataLabel = labelEncoder.fit_transform(Y_train[1:,])

predict = kNearestNeighbors(2, titanicTrainingData, titanicTrainingDataLabel, X_testArray)
print('Predict:', predict)

# (numpy.ndarray.astype)
#   Copy of the array, cast to a specified type.
predictRev = labelEncoder.inverse_transform(predict.astype(int))
print('Predict Revision:', predictRev)

#%%
kaggleSubmission = dataFrame_test[['PassengerId']]
kaggleSubmission['Survived'] = predictRev
kaggleSubmission.to_csv(dataDir + 'kaggleSubmission.csv', index=False)
