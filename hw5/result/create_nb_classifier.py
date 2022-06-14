import sys
import os
import pandas
import csv
from sklearn import svm, preprocessing, neighbors, ensemble, neural_network, naive_bayes
import joblib


def create_nb_classifier(joblib_file):
    '''
        Train Naive Bayes classifier with training set (feature vectors) from csv file,
        and load the trained model in .joblib file
    '''    
    classifier = naive_bayes.GaussianNB()
    classifier.fit(x_train, y_train) # fit svm classifier to the train data
    y_prediction = classifier.predict(x_test)
    
    print("Naive Bayes classifier: ")
    print("Accuracy on train set: ", classifier.score(x_train, y_train))
    print("Accuracy on test set: ", classifier.score(x_test, y_test))
    
    joblib.dump(classifier, joblib_file) # save classifier in the joblib file


training_data = pandas.read_csv('train_test_datasets/features_train_70000.csv') #import our training data from the csv file (56,000 image)
testing_data = pandas.read_csv('train_test_datasets/features_test_70000.csv') #import our test data from the csv file (14,000 image)

x_train, y_train = training_data.drop([' Tag'], axis=1), training_data[' Tag']
x_test, y_test = testing_data.drop([' Tag'], axis=1), testing_data[' Tag']

scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
x_train, x_test = scaler.fit_transform(x_train), scaler.fit_transform(x_test) #scale the features in the interval [0:1]


create_nb_classifier('nb_classifier.joblib')