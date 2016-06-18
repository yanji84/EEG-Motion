import numpy as np
import pandas as pd
import features_extract as fe
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import metrics

# which subject to train/test on
subject = 1

# extract features from train data
subj_series1_data = pd.read_csv('train/subj' + str(subject) + '_series1_data.csv', header=None, sep=',')
subj_series1_labels = pd.read_csv('train/subj' + str(subject) + '_series1_events.csv', header=None, sep=',')
train_data = subj_series1_data[1:][range(1,33)].values.astype(float)
train_features = fe.extractFeatures_by_sliding_window(train_data)

# extract labels for the train data
labels = []
labels.append(subj_series1_labels[1:][1].values.astype(float))
labels.append(subj_series1_labels[1:][2].values.astype(float))
labels.append(subj_series1_labels[1:][3].values.astype(float))
labels.append(subj_series1_labels[1:][4].values.astype(float))
labels.append(subj_series1_labels[1:][5].values.astype(float))
labels.append(subj_series1_labels[1:][6].values.astype(float))

# extract features from test data
subj_series9_data = pd.read_csv('test/subj' + str(subject) + '_series9_data.csv', header=None, sep=',')
subj_series10_data = pd.read_csv('test/subj' + str(subject) + '_series10_data.csv', header=None, sep=',')
series9 = subj_series9_data[1:][range(1,33)].values.astype(float)
series10 = subj_series10_data[1:][range(1,33)].values.astype(float)
test_data = np.concatenate((series9,series10))
test_features = fe.extractFeatures_by_sliding_window(test_data)

# train one model for each class and predict probablity of each class for every test data
predicted = []
for clas in range(0, 6):
	model = LogisticRegression(penalty='l2')
	model.fit(train_features, labels[clas])
	predicted.append([model.predict_proba(test_feature)[0][1] for test_feature in test_features])

# output the results to csv
id1 = np.array(subj_series9_data[1:][0].values.astype(str))
id2 = np.array(subj_series10_data[1:][0].values.astype(str))
id_all = np.concatenate((id1, id2))
cols = ['HandStart','FirstDigitTouch','BothStartLoadPhase','LiftOff','Replace','BothReleased']
submission = pd.DataFrame(index=id_all, columns=cols, data=np.array(predicted).T)
submission.to_csv('w207.csv',index_label='id',float_format='%.3f') 