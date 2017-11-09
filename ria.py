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
#print(labels.value_counts())



r,c = training.shape



training.as_matrix()

print(training.shape)


for i in range(r):
	print training[i,c-1]
	if(training[i,c-1]=='5'):
		subset1_features.append(training[i,0:c-2])
		subset1_label.append(training[i,c-1])



#making 10 subsets of training data


























































unit = r/10
subset1_features = training_features[0:unit]
print(subset_features.type)
subset1_label = training_label[0:unit]
print(subset1_label.shape)


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




