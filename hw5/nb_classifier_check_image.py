import matplotlib.pyplot as plt
import cv2
from numpy import std, ptp, median
from scipy.stats import skew, kurtosis, mstats
import eeglib
import joblib
from sklearn import preprocessing
import pandas as pd
import os

import warnings
warnings.filterwarnings("ignore")


def stego_or_clean(img_path, classifier):

    im = cv2.imread(img_path)
    vals = im.mean(axis=2).flatten()
    # plot histogram with 255 bins
    b, bins, patches = plt.hist(vals, 255)
    data = {'Kurtosis' : [kurtosis(b)], 
        ' Skewness' : [skew(b)], 
        ' Std' : [std(b)], 
        ' Range' : [ptp(b)],
        ' Median': [median(b)],
        ' Geometric_Mean': [mstats.gmean(b)],
        ' Mobility': [eeglib.features.hjorthMobility(b)],
        ' Complexity': [eeglib.features.hjorthComplexity(b)]
       }
    df = pd.DataFrame(data)
    print(df)
    testing_data = pd.read_csv('train_test_datasets/features_test_70000.csv')
    x_test, y_test = testing_data.drop([' Tag'], axis=1), testing_data[' Tag']
    x_test = x_test.append(df, ignore_index=True)
    
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    x_test = scaler.fit_transform(x_test)
    
    result = 'stego' if classifier.predict([x_test[13999]])[0] == 1 else 'clean'

    return result


nb_classifier = joblib.load('nb_classifier.joblib')
image_path = 'images/image3_with_watermark3.jpg'
result = stego_or_clean(image_path, nb_classifier)
print(result)