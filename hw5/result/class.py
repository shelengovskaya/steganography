import math
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
import operator
import seaborn as sns
from sklearn import datasets, metrics, model_selection


def mean(samples):
    return sum(samples) / float(len(samples))

def stdev(samples):
    avg = mean(samples)
    variance = sum([pow(x - avg, 2) for x in samples]) / float(len(samples) - 1)
    return math.sqrt(variance)

def fit_one_class(instances):
    return [(mean(attribute), stdev(attribute)) for attribute in zip(*instances)]

def split_by_classes(X, y):
    separated = {}
    for features, cls in zip(X, y):
        if cls not in separated:
            separated[cls] = []
        separated[cls].append(features)
    return separated

def fit(X, y):
    summaries = {}
    separated = split_by_classes(X, y)
    for cls, instances in separated.items():
        summaries[cls] = fit_one_class(instances)
    return summaries

# different probability functions
def gaussian_probability(x, mean_value, stdev_value):
    """
    Gaussian Probability Density Function
    """
    exponent = math.exp(-(math.pow(x - mean_value, 2) / (2 * math.pow(stdev_value, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev_value)) * exponent

# TODO: multinomial, bernoulli or kernel naive bayes

def class_probability(summary, features, prob_fn):
    is_array_of_features = True
    try:
        len(features)
    except TypeError:
        is_array_of_features = False
        
    if is_array_of_features:
        return [class_probability_single(summary, features_raw, prob_fn) for features_raw in features]
    else:
        return [class_probability_single(summary, features, prob_fn)]

def class_probability_single(summary, features, prob_fn):
    probabilities = {}
    for cls, cls_summary in summary.items():
        res = 1
        for feature, (mean_value, stdev_value) in zip(features, cls_summary):
            res *= prob_fn(feature, mean_value, stdev_value)
        probabilities[cls] = res
    return probabilities

def predict(summary, X, prob_fn):
    prob_of_classes = class_probability(summary, X, prob_fn)
    return [max(p.items(), key=operator.itemgetter(1))[0] for p in prob_of_classes]

def precision(x_pred, y_true):
    return sum(y_pred == y_test) / len(y_test)

def macro_recall(x_pred, y_true):
    """
    # recall, average='macro'
    # Calculate metrics for each label, and find their unweighted mean. 
    # This does not take label imbalance into account.
    """    
    y_pred_array = np.array(y_pred)
    recall_one_cls = []
    for cls in set(y_test):
        tp = sum(y_true[y_pred_array == cls] == cls)
        fn = sum(y_true[y_pred_array != cls] == cls)
        recall_one_cls.append(tp / (tp + fn))   
    return sum(recall_one_cls) / len(recall_one_cls)



# random_state = np.random.RandomState(0)
# data = datasets.load_wine()

file = open("train_test_datasets/features_train_70000.csv")
X = np.loadtxt(file, delimiter=",")

X_train = X[:, :-1]
y_train = X[:,-1]

file = open("train_test_datasets/features_test_70000.csv")
X = np.loadtxt(file, delimiter=",")

X_test = X[:, :-1]
y_test = X[:,-1]


# X_train, X_test, y_train, y_test = model_selection.train_test_split(
#     data.data, data.target, test_size=0.3, random_state=random_state)



# train model
model = fit(X_train, y_train)
y_pred = predict(model, X_test, gaussian_probability)

print('accuracy:')
print(precision(y_pred, y_test))
print('recall:')
print(macro_recall(y_pred, y_test))
print('confusion matrix:')
print(metrics.confusion_matrix(y_pred, y_test))
