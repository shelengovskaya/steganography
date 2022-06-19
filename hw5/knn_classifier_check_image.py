import numpy as np
from scipy.stats import mode
import pandas as pd
from sklearn.metrics import accuracy_score
 

def eucledian(p1, p2):
    dist = np.sqrt(np.sum((p1-p2)**2))
    return dist
 

def predict(x_train, y , x_input, k):

    data = {'Kurtosis' : [x_input[0][0]], 
        ' Skewness' : [x_input[0][1]], 
        ' Std' : [x_input[0][2]], 
        ' Range' : [x_input[0][3]],
        ' Median': [x_input[0][4]],
        ' Geometric_Mean': [x_input[0][5]],
        ' Mobility': [x_input[0][6]],
        ' Complexity': [x_input[0][7]]
       }
    df = pd.DataFrame(data)
    print(df)

    op_labels = []
     
    for item in x_input: 
        point_dist = []

        for j in range(len(x_train)): 
            distances = eucledian(np.array(x_train[j,:]) , item) 
            point_dist.append(distances) 
        point_dist = np.array(point_dist) 
         
        dist = np.argsort(point_dist)[:k] 
        labels = y[dist]
        lab = mode(labels) 
        lab = lab.mode[0]
        op_labels.append(lab)

 
    return op_labels


file = open("train_test_datasets/updated_features_train_70000.csv")
X = np.loadtxt(file, delimiter=",")

X_train = X[:, :-1]
y_train = X[:,-1]


test_count = 3

file = open("train_test_datasets/updated_features_test_70000.csv")
X = np.loadtxt(file, delimiter=",")

X_test = X[:test_count, :-1]
y_test = X[:test_count,-1]


y_pred = predict(X_train, y_train, X_test, 7)
result = ['stego' if int(r) == 1 else 'clean' for r in y_pred]

print(result)

accuracy = accuracy_score(y_test, y_pred)

print('Accuracy:', accuracy)
