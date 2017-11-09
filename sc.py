import csv
import random
import math

import pandas
import numpy as np
from time import time
from sklearn import tree
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from nb import *
#from _future_ import division


col_names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]


num_features = ["duration","protocol_type","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files",
    "is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate"]




#testing dataset preperation


testing= pandas.read_csv("testing", header=None, names = col_names)
# print "Testing dataset shape with label"
# print(testing.shape)

labels = testing['protocol_type']
labels[labels=='icmp'] = 1
labels[labels=='tcp'] = 2
labels[labels=='udp'] = 3

labels = testing['flag']
labels[labels=='SF'] = 1
labels[labels=='REJ'] = 2
labels[labels=='S0'] = 3
labels[labels=='RSTO'] = 4
labels[labels=='RSTR'] = 5
labels[labels=='S3'] = 6
labels[labels=='SH'] = 7
labels[labels=='S1'] = 8
labels[labels=='S2'] = 9
labels[labels=='OTH'] = 10
labels[labels=='RSTOS0'] = 11

labels = testing['label']
labels[labels=='back.'] = 'dos'
labels[labels=='buffer_overflow.'] = 'u2r'
labels[labels=='ftp_write.'] = 'r2l'
labels[labels=='guess_passwd.'] = 'r2l'
labels[labels=='imap.'] = 'r2l'
labels[labels=='ipsweep.'] = 'probe'
labels[labels=='land.'] = 'dos'
labels[labels=='loadmodule.'] = 'u2r'
labels[labels=='multihop.'] = 'r2l'
labels[labels=='neptune.'] = 'dos'
labels[labels=='nmap.'] = 'probe'
labels[labels=='perl.'] = 'u2r'
labels[labels=='phf.'] = 'r2l'
labels[labels=='pod.'] = 'dos'
labels[labels=='portsweep.'] = 'probe'
labels[labels=='rootkit.'] = 'u2r'
labels[labels=='satan.'] = 'probe'
labels[labels=='smurf.'] = 'dos'
labels[labels=='spy.'] = 'r2l'
labels[labels=='teardrop.'] = 'dos'
labels[labels=='warezclient.'] = 'r2l'
labels[labels=='warezmaster.'] = 'r2l'
labels[labels=='saint.'] = 'probe'
labels[labels=='mscan.'] = 'probe'
labels[labels=='apache2.'] = 'dos'
labels[labels=='udpstorm.'] = 'dos'
labels[labels=='processtable.'] = 'dos'
labels[labels=='mailbomb.'] = 'dos'
labels[labels=='xterm.'] = 'u2r'
labels[labels=='ps.'] = 'u2r'
labels[labels=='sqlattack.'] = 'u2r'
labels[labels=='snmpgetattack.'] = 'r2l'
labels[labels=='named.'] = 'r2l'
labels[labels=='xlock.'] = 'r2l'
labels[labels=='xsnoop.'] = 'r2l'
labels[labels=='sendmail.'] = 'r2l'
labels[labels=='httptunnel.'] = 'r2l'
labels[labels=='worm.'] = 'r2l'
labels[labels=='snmpguess.'] = 'r2l'

# print "types of attacks in testing dataset and  their percentage"
# print(labels.value_counts()/311029*100)

labels = testing['label']
labels[labels=='dos'] = '1'
labels[labels=='normal.'] = '2'
labels[labels=='probe'] = '3'
labels[labels=='r2l'] = '4'
labels[labels=='u2r'] = '5'

# print "types of attacks in training dataset and  their percentage"
# print(labels.value_counts()/311029*100)

testing_features = testing[num_features]
# print "testing data features shape",testing_features.shape

testing_label = testing['label']
# print "testing data label shape",testing_label.shape





#traing data preperation

training = pandas.read_csv("training", header=None, names = col_names)
# print "Training dataset shape with label"
# print(training.shape)

labels = training['protocol_type']
labels[labels=='icmp'] = 1
labels[labels=='tcp'] = 2
labels[labels=='udp'] = 3

labels = training['flag']
labels[labels=='SF'] = 1
labels[labels=='REJ'] = 2
labels[labels=='S0'] = 3
labels[labels=='RSTO'] = 4
labels[labels=='RSTR'] = 5
labels[labels=='S3'] = 6
labels[labels=='SH'] = 7
labels[labels=='S1'] = 8
labels[labels=='S2'] = 9
labels[labels=='OTH'] = 10
labels[labels=='RSTOS0'] = 11


labels = training['label']
labels[labels=='back.'] = 'dos'
labels[labels=='buffer_overflow.'] = 'u2r'
labels[labels=='ftp_write.'] = 'r2l'
labels[labels=='guess_passwd.'] = 'r2l'
labels[labels=='imap.'] = 'r2l'
labels[labels=='ipsweep.'] = 'probe'
labels[labels=='land.'] = 'dos'
labels[labels=='loadmodule.'] = 'u2r'
labels[labels=='multihop.'] = 'r2l'
labels[labels=='neptune.'] = 'dos'
labels[labels=='nmap.'] = 'probe'
labels[labels=='perl.'] = 'u2r'
labels[labels=='phf.'] = 'r2l'
labels[labels=='pod.'] = 'dos'
labels[labels=='portsweep.'] = 'probe'
labels[labels=='rootkit.'] = 'u2r'
labels[labels=='satan.'] = 'probe'
labels[labels=='smurf.'] = 'dos'
labels[labels=='spy.'] = 'r2l'
labels[labels=='teardrop.'] = 'dos'
labels[labels=='warezclient.'] = 'r2l'
labels[labels=='warezmaster.'] = 'r2l'

# print "types of attacks in training dataset and  their percentage"
# print(labels.value_counts()/494021*100)

labels = training['label']
labels[labels=='dos'] = '1'
labels[labels=='normal.'] = '2'
labels[labels=='probe'] = '3'
labels[labels=='r2l'] = '4'
labels[labels=='u2r'] = '5'

# print "types of attacks in training dataset and  their percentage"
# print(labels.value_counts()/494021*100)

training_features = training[num_features]
# print "training data features shape",training_features.shape

training_label = training['label']
# print "training data label shape",training_label.shape
print(labels.value_counts())





#making 10 subsets of training data

unit = 494021/10
subset1_features = training_features[0:unit]
subset1_label = training_label[0:unit]


subset2_features = training_features[unit:2*unit]
subset2_label = training_label[unit:2*unit]

subset3_features = training_features[2*unit:3*unit]
subset3_label = training_label[2*unit:3*unit]

subset4_features = training_features[3*unit:4*unit]
subset4_label = training_label[3*unit:4*unit]

subset5_features = training_features[4*unit:5*unit]
subset5_label = training_label[4*unit:5*unit]

subset6_features = training_features[5*unit:6*unit]
subset6_label = training_label[5*unit:6*unit]

subset7_features = training_features[6*unit:7*unit]
subset7_label = training_label[6*unit:7*unit]

subset8_features = training_features[7*unit:8*unit]
subset8_label = training_label[7*unit:8*unit]

subset9_features = training_features[8*unit:9*unit]
subset9_label = training_label[8*unit:9*unit]

subset10_features = training_features[9*unit:10*unit]
subset10_label = training_label[9*unit:10*unit]



#training 10 tree classiefiers

clf1 = tree.DecisionTreeClassifier()
clf1.fit(subset1_features,subset1_label)


clf2 = tree.DecisionTreeClassifier()
clf2.fit(subset2_features,subset2_label)

clf3 = tree.DecisionTreeClassifier()
clf3.fit(subset3_features,subset3_label)

clf4 = tree.DecisionTreeClassifier()
clf4.fit(subset4_features,subset4_label)

clf5 = tree.DecisionTreeClassifier()
clf5.fit(subset5_features,subset5_label)

clf6 = tree.DecisionTreeClassifier()
clf6.fit(subset6_features,subset6_label)

clf7 = tree.DecisionTreeClassifier()
clf7.fit(subset7_features,subset7_label)

clf8 = tree.DecisionTreeClassifier()
clf8.fit(subset8_features,subset8_label)

clf9 = tree.DecisionTreeClassifier()
clf9.fit(subset9_features,subset9_label)

clf10 = tree.DecisionTreeClassifier()
clf10.fit(subset10_features,subset10_label)





#predicting on training dataset

# print "following are predictions from 10 classifiers on whole training dataset"
pred1 = clf1.predict(training_features)
# print(pred1)

pred2 = clf2.predict(training_features)
# print(pred2)

pred3 = clf3.predict(training_features)
# print(pred3)

pred4 = clf4.predict(training_features)
# print(pred4)

pred5 = clf5.predict(training_features)
# print(pred5)

pred6 = clf6.predict(training_features)
# print(pred6)

pred7 = clf7.predict(training_features)
# print(pred7)

pred8 = clf8.predict(training_features)
# print(pred8)

pred9 = clf9.predict(training_features)
# print(pred9)

pred10 = clf10.predict(training_features)
# print(pred10)




#making a table of all training predictions

table = np.empty((494021,10),dtype='float')
table[0:: ,0] = pred1
table[0:: ,1] = pred2
table[0:: ,2] = pred3
table[0:: ,3] = pred4
table[0:: ,4] = pred5
table[0:: ,5] = pred6
table[0:: ,6] = pred7
table[0:: ,7] = pred8
table[0:: ,8] = pred9
table[0:: ,9] = pred10
#table[0:: ,10] = training_label

table2 = np.empty((494021,11),dtype='float')
table2[0:: ,0] = pred1
table2[0:: ,1] = pred2
table2[0:: ,2] = pred3
table2[0:: ,3] = pred4
table2[0:: ,4] = pred5
table2[0:: ,5] = pred6
table2[0:: ,6] = pred7
table2[0:: ,7] = pred8
table2[0:: ,8] = pred9
table2[0:: ,9] = pred10
table2[0:: ,10] = training_label

# print "combined table storing results of all 10 predictors on training dataset"
# print(table.shape)
# print(table[0:: ,0::])


#Training naive bias classifier

gnb = GaussianNB()
gnb.fit(table,training_label)







#predictions for testing data using tree classifier

# print "following are predictions from 10 classifiers on whole testing dataset"

pred1 = clf1.predict(testing_features)
# print(pred1)

pred2 = clf2.predict(testing_features)
# print(pred2)

pred3 = clf3.predict(testing_features)
# print(pred3)

pred4 = clf4.predict(testing_features)
# print(pred4)

pred5 = clf5.predict(testing_features)
# print(pred5)

pred6 = clf6.predict(testing_features)
# print(pred6)

pred7 = clf7.predict(testing_features)
# print(pred7)

pred8 = clf8.predict(testing_features)
# print(pred8)

pred9 = clf9.predict(testing_features)
# print(pred9)

pred10 = clf10.predict(testing_features)
# print(pred10)


#making a table of all training predictions

table = np.empty((311029,10),dtype='float')
table[0:: ,0] = pred1
table[0:: ,1] = pred2
table[0:: ,2] = pred3
table[0:: ,3] = pred4
table[0:: ,4] = pred5
table[0:: ,5] = pred6
table[0:: ,6] = pred7
table[0:: ,7] = pred8
table[0:: ,8] = pred9
table[0:: ,9] = pred10
#table[0:: ,10] = testing_label

print "combined table storing results of all 10 predictors on testing dataset"
# print(table.shape)
# print(table[0:: ,0::])

#predictions naive bias classifier

predictions = gnb.predict(table)
print "final predictions on testing data using NB"
#print(predictions)

#finding accuracy
print "accuracy"
acc = accuracy_score(testing_label,predictions)
#print(acc*100)



table3 = np.empty((311029,11),dtype='float')
table3[0:: ,0] = pred1
table3[0:: ,1] = pred2
table3[0:: ,2] = pred3
table3[0:: ,3] = pred4
table3[0:: ,4] = pred5
table3[0:: ,5] = pred6
table3[0:: ,6] = pred7
table3[0:: ,7] = pred8
table3[0:: ,8] = pred9
table3[0:: ,9] = pred10
table3[0:: ,10] = testing_label




#print(100*naivebayes(table2,table3))
print(acc*100)

#generating confusion matrix using predictions and labels
print "calculations using confusion matrix"
confusion = confusion_matrix(predictions,testing_label)
#print(confusion)

total = 311029
re = np.empty((5,4),dtype='float')
t2 = np.empty((5,3),dtype='float')


re[0][0]=confusion[0][0];
re[0][1]=confusion[0][1]+confusion[0][2]+confusion[0][3]+confusion[0][4];
re[0][2]=confusion[1][0]+confusion[2][0]+confusion[3][0]+confusion[4][0];
re[0][3]=total - (re[0][0]+re[0][1]+re[0][2]);

re[1][0]=confusion[1][1];
re[1][1]=confusion[1][0]+confusion[1][2]+confusion[1][3]+confusion[1][4];
re[1][2]=confusion[0][1]+confusion[2][1]+confusion[3][1]+confusion[4][1];
re[1][3]=total - (re[1][0]+re[1][1]+re[1][2]);

re[2][0]=confusion[2][2];
re[2][1]=confusion[2][0]+confusion[2][1]+confusion[2][3]+confusion[2][4];
re[2][2]=confusion[0][2]+confusion[1][2]+confusion[3][2]+confusion[4][2];
re[2][3]=total - (re[2][0]+re[2][1]+re[2][2]);

re[3][0]=confusion[3][3];
re[3][1]=confusion[3][0]+confusion[3][1]+confusion[3][2]+confusion[3][4];
re[3][2]=confusion[0][3]+confusion[1][3]+confusion[2][3]+confusion[4][3];
re[3][3]=total - (re[3][0]+re[3][1]+re[3][2]);

re[4][0]=confusion[4][4];
re[4][1]=confusion[4][0]+confusion[4][2]+confusion[4][3]+confusion[4][1];
re[4][2]=confusion[1][4]+confusion[2][4]+confusion[3][4]+confusion[1][4];
re[4][3]=total - (re[4][0]+re[4][1]+re[4][2]);


t2[0][0]=(re[0][0]+re[0][2])/(re[0][0]+re[0][1]+re[0][2]+re[0][3]);
t2[0][1]=re[0][0]/(re[0][0]+re[0][3]);
t2[0][2]=re[0][1]/(re[0][1]+re[0][2]);

t2[1][0]=(re[1][0]+re[1][2])/(re[1][0]+re[1][1]+re[1][2]+re[1][3]);
t2[1][1]=re[1][0]/(re[1][0]+re[1][3]);
t2[1][2]=re[1][1]/(re[1][1]+re[1][2]);

t2[2][0]=(re[2][0]+re[2][2])/(re[2][0]+re[2][1]+re[2][2]+re[2][3]);
t2[2][1]=re[2][0]/(re[2][0]+re[2][3]);
t2[2][2]=re[2][1]/(re[2][1]+re[2][2]);

t2[3][0]=(re[3][0]+re[3][2])/(re[3][0]+re[3][1]+re[3][2]+re[3][3]);
t2[3][1]=re[3][0]/(re[3][0]+re[3][3]);
t2[3][2]=re[3][1]/(re[3][1]+re[3][2]);

t2[4][0]=(re[4][0]+re[4][2])/(re[4][0]+re[4][1]+re[4][2]+re[4][3]);
t2[4][1]=re[4][0]/(re[4][0]+re[4][3]);
t2[4][2]=re[4][1]/(re[4][1]+re[4][2]);

#print(t2*100)

