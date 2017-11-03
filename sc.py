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

training = pandas.read_csv("training", header=None, names = col_names)
print "Training dataset shape with label"
print(training.shape)

testing= pandas.read_csv("testing", header=None, names = col_names)
print "Testing dataset shape with label"
print(testing.shape)



num_features = ["duration","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files",
    "is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate"]


#print(training['label'].value_counts())

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


training_features = training[num_features]
training_label = training['label']
print "training data features shape",training_features.shape
print "training data label shape",training_label.shape

#print "types of attacks in training dataset and  their percentage"
#print(labels.value_counts()/494021*100)

#print(testing['label'].value_counts())


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

#print "types of attacks in testing dataset and  their percentage"
#print(labels.value_counts()/311029*100)


testing_features = testing[num_features]
testing_label = testing['label']
print "testing data features shape",testing_features.shape
print "testing data label shape",testing_label.shape






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



print(training['label'].value_counts())
print(testing['label'].value_counts())



pred1 = clf1.predict(subset1_features)
print(pred1)

pred2 = clf2.predict(subset2_features)
print(pred2)

pred3 = clf3.predict(subset3_features)
print(pred3)

pred4 = clf4.predict(subset4_features)
print(pred4)

pred5 = clf5.predict(subset5_features)
print(pred5)

pred6 = clf6.predict(subset6_features)
print(pred6)

pred7 = clf7.predict(subset7_features)
print(pred7)

pred8 = clf8.predict(subset8_features)
print(pred8)

pred9 = clf9.predict(subset9_features)
print(pred9)

pred10 = clf10.predict(subset10_features)
print(pred10)


table = np.empty((49402,10),dtype='string')
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


print(table.shape)
print(table[0:: ,0::])

for i in range(49402):
	for j in range(10):
		if(table[i,j]=='d'):
			table[i,j] = 1

		if(table[i,j]=='n'):
			table[i,j] = 2

		if(table[i,j]=='p'):
			table[i,j] = 3

		if(table[i,j]=='r'):
			table[i,j] = 4

		if(table[i,j]=='u'):
			table[i,j] = 5


table.astype(np.float)
print(table.shape)
print(table[0:: ,0::])



for i in range(494021):
		if(training_label[i]=='dos'):
			training_label[i] = 1

		if(training_label[i]=='normal.'):
			training_label[i] = 2

		if(training_label[i]=='probe'):
			training_label[i] = 3

		if(training_label[i]=='r2l'):
			training_label[i] = 4

		if(training_label[i]=='u2r'):
			training_label[i] = 5

training_label.astype(np.float)
print(training_label.shape)
print(training_label)


gnb = GaussianNB()
gnb.fit(table,training_label)











