import numpy as np
import pandas as pd
import features_extract as fe
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import metrics

# specify which class to train for
class_to_test = 0
# specify which subject to train for
subject = 1
# specify which series to use for training
train_series = 1
# specify which series to use for testing
test_series = 8

# extract features from train data
train_data = pd.read_csv('train/subj' + str(subject) + '_series' + str(train_series) + '_data.csv', header=None, sep=',')
train_labels = pd.read_csv('train/subj' + str(subject) + '_series' + str(train_series) + '_events.csv', header=None, sep=',')
train_data = train_data[1:][range(1,33)].values.astype(float)
train_features = fe.extractFeatures_by_sliding_window(train_data)

# extract labels from train data
labels = []
labels.append(train_labels[1:][1].values.astype(float))
labels.append(train_labels[1:][2].values.astype(float))
labels.append(train_labels[1:][3].values.astype(float))
labels.append(train_labels[1:][4].values.astype(float))
labels.append(train_labels[1:][5].values.astype(float))
labels.append(train_labels[1:][6].values.astype(float))

# extract features from test data
test_data = pd.read_csv('train/subj' + str(subject) + '_series' + str(test_series) + '_data.csv', header=None, sep=',')
test_labels = pd.read_csv('train/subj' + str(subject) + '_series' + str(test_series) + '_events.csv', header=None, sep=',')
test_data = test_data[1:][range(1,33)].values.astype(float)
test_features = fe.extractFeatures_by_sliding_window(test_data)

# extract labels from test data
t_labels = []
t_labels.append(test_labels[1:][1].values.astype(float))
t_labels.append(test_labels[1:][2].values.astype(float))
t_labels.append(test_labels[1:][3].values.astype(float))
t_labels.append(test_labels[1:][4].values.astype(float))
t_labels.append(test_labels[1:][5].values.astype(float))
t_labels.append(test_labels[1:][6].values.astype(float))

# train a model on the specified class
model = LogisticRegression(penalty='l2')
model.fit(train_features, labels[class_to_test])

# print auc
predicted = [model.predict(test_feature)[0] for test_feature in test_features]
predicted = np.array(predicted)
fpr, tpr, thresholds = metrics.roc_curve(t_labels[class_to_test], predicted, pos_label=1)
print(metrics.auc(fpr, tpr))