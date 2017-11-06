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

from nb import *
from sc import *

print(confusion)

tdos = confusion[0][0]
tnormal = confusion[1][1]
tprobe = confusion[2][2]
tr2l = confusion[3][3]
tu2r = confusion[4][4]


fdos = confusion[0][1] + confusion[0][2] + confusion[0][3] + confusion[0][4]
fnormal = confusion[1][0] + confusion[1][2] + confusion[1][3] + confusion[1][4]
fprobe = confusion[2][0] + confusion[2][1] + confusion[2][3] + confusion[2][4]
fr2l = confusion[3][0] + confusion[3][1] + confusion[3][2] + confusion[3][4]
fu2r = confusion[4][0] + confusion[4][1] + confusion[4][2] + confusion[4][3]

conf_acc = (tdos + tnormal + tprobe + tr2l + tu2r)
conf_acc = conf_acc*100/311029
conf_acc = float(conf_acc)
print(conf_acc)
