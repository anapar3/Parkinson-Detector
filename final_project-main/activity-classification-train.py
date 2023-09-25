# -*- coding: utf-8 -*-
"""
This is the script used to train an activity recognition 
classifier on accelerometer data.

"""

import os
import sys
import numpy as np
import sklearn 
from sklearn.tree import export_graphviz , DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from features import extract_features
from util import slidingWindow, reorient, reset_vars
import pickle
import matplotlib.pyplot as plt

import labels


# %%---------------------------------------------------------------------------
#
#		                 Load Data From Disk
#
# -----------------------------------------------------------------------------

print("Loading data...")
sys.stdout.flush()
data_file = 'data/All-Activities/all_labeled_data_holding.csv'
data = np.genfromtxt(data_file, delimiter=',')
print("Loaded {} raw labelled activity data samples.".format(len(data)))
sys.stdout.flush()

# %%---------------------------------------------------------------------------
#
#		                    Pre-processing
#
# -----------------------------------------------------------------------------

print("Reorienting accelerometer data...")
sys.stdout.flush()
reset_vars()
reoriented = np.asarray([reorient(data[i,2], data[i,3], data[i,4]) for i in range(len(data))])
reoriented_data_with_timestamps = np.append(data[:,0:2],reoriented,axis=1)
data = np.append(reoriented_data_with_timestamps, data[:,-1:], axis=1)

data = np.nan_to_num(data)

# %%---------------------------------------------------------------------------
#
#		                Extract Features & Labels
#
# -----------------------------------------------------------------------------

window_size = 20
step_size = 20

# sampling rate should be about 100 Hz (sensor logger app); you can take a brief window to confirm this
n_samples = 1000
time_elapsed_seconds = (data[n_samples,1] - data[0,1])
sampling_rate = n_samples / time_elapsed_seconds

print("Sampling Rate: " + str(sampling_rate))

# TODO: list the class labels that you collected data for in the order of label_index (defined in labels.py)
class_names = labels.activity_labels

print("Extracting features and labels for window size {} and step size {}...".format(window_size, step_size))
sys.stdout.flush()

X = []
Y = []
feature_names = []
for i,window_with_timestamp_and_label in slidingWindow(data, window_size, step_size):
    window = window_with_timestamp_and_label[:,2:-1]
    # print("window = ")
    # print(window)
    feature_names, x = extract_features(window)
    # print(feature_names , x)
    
    X.append(x)
    Y.append(window_with_timestamp_and_label[10, -1])
    
X = np.asarray(X)
Y = np.asarray(Y)
# print("X: " , X.shape())
# print("Y: " , Y.shape())
n_features = len(X)
    
print("Finished feature extraction over {} windows".format(len(X)))
print("Unique labels found: {}".format(set(Y)))
print("\n")
sys.stdout.flush()

# %%---------------------------------------------------------------------------
#
#		                Train & Evaluate Classifier
#
# -----------------------------------------------------------------------------


# TODO: split data into train and test datasets using 10-fold cross validation
cv = KFold(n_splits=10, random_state=None, shuffle=True)

fold_data = cv.split(data)

tree = DecisionTreeClassifier(criterion="entropy", max_depth=3)

sum_accuracy = 0
sum_recall = 0
ssum_precision = 0

def get_evaluation(confusion_matrix):
    tp = []
    fp = []
    fn = [0,0]

    for i in range (len(confusion_matrix)):
        temp_fp = 0
        for j in range (len(confusion_matrix[0])):
            if(i == j):
                tp.append(confusion_matrix[i][j])
            else:
                temp_fp+= confusion_matrix[i][j]
        fp.append(temp_fp)
    
    for j in range (len(confusion_matrix[0])):
        for i in range (len(confusion_matrix)):
            # print("r_iter:" , confusion_matrix[i][j])
            if(i != j):
                fn[j] += confusion_matrix[i][j]

    acc = sum(tp)/ (sum(tp) + sum(fp))

    prec = [0,0]
    rec = [0,0]

    for i in range (len(prec)):
        prec[i] = tp[i]/(tp[i] + fp[i])
        rec[i] =  tp[i]/(tp[i] + fn[i])

    return acc , prec , rec
    

avg_acc = 0
avg_prec = [0 , 0]
avg_rec = [0 , 0]

for i, (train_index , test_index) in enumerate(fold_data):
    print("Fold: " , i)
    print("Train index: " , train_index)
    print("Test index: " , test_index)

    train_x = []
    train_y = []
    
    test_x = []
    test_y = []

    assert(len(train_x) == len(train_y))

    for j in train_index:
        if(j < len(X)):
            train_x.append(X[j])
            train_y.append(Y[j])

    for k in test_index:
        if (k < len(X)):
            test_x.append(X[k])
            test_y.append(Y[k])

    # print("TrainX = " , train_x)
    # print("TrainY = " , train_y)
    tree.fit(train_x , train_y)
    print("Fit done")
    pred_y = tree.predict(test_x)
    print("Predcit done")
    conf = confusion_matrix(test_y, pred_y)
    print("matrix done")
    print(conf)
    acc , prec , rec = get_evaluation(conf)
    print("acc: " , acc)
    print("prec: " , prec)
    print("rec: " , rec)
    avg_acc += acc
    for k in range (len(avg_prec)):
        avg_prec[k] += prec[k]
        avg_rec[k] += rec[k]

print("avg acc: " , avg_acc/10)
for j in range (len(avg_prec)):
    avg_prec[j] = avg_prec[j]/10
    avg_rec[j] = avg_rec[j]/10
print("avg rec: " , avg_prec)
print("avg prec: " , avg_rec)



"""
TODO: iterating over each fold, fit a decision tree classifier on the training set.
Then predict the class labels for the test set and compute the confusion matrix
using predicted labels and ground truth values. Print the accuracy, precision and recall
for each fold.
"""

# TODO: calculate and print the average accuracy, precision and recall values over all 10 folds


# TODO: train the decision tree classifier on entire dataset


# TODO: Save the decision tree visualization to disk - replace 'tree' with your decision tree and run the below line
export_graphviz(tree, out_file='tree.dot', feature_names = feature_names)

# TODO: Save the classifier to disk - replace 'tree' with your decision tree and run the below line
print("saving classifier model...")
with open('classifier2.pickle', 'wb') as f:
    pickle.dump(tree, f)

