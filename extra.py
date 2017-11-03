#clf = ExtraTreesClassifier()
#clf = clf.fit(train_features,train_label)
#importance = clf.feature_importances_
 
#model = SelectFromModel(clf, prefit=True)
#train_features = model.transform(train_features)
#print "training features shape after feature selection"
#print(train_features.shape) 


#labels = kdd_data_10percent['label'].copy()
#labels[labels!='normal.'] = 'attack.'
#labels.value_counts()

#df = pandas.DataFrame(train_features)
#print(df.shape)
#df['label'] = labels
#print(df.shape)
#new_data = df.as_matrix()
#print(df['label'].value_counts())







temp_label = temp_data['label']
	temp_label = temp_label.as_matrix()
	temp_data = temp_data.as_matrix()
	
	
	
	train_pred=clf[i].predict(temp_features)
	gnb.fit(train_pred,train_label)


pred = []
for i in range(10):
	pred[i] = clf[i].predict(test_features)


gnb = GaussianNB()
pred2 = gnb.predict(pred)
acc = accuracy_score(test_label,pred2)
print(acc)

unit = train_rows/10
subset_data = np.array([10,unit,train_cols])
for i in range(10):
	temp_data = train_data[i:i+unit]
	np.append(subset_data,temp_data)
	print "subset_data ",i," shape ",subset_data.shape

subset_features= np.array([10,unit,train_cols-1])
for i in range(10):
	temp_features = subset_data[i,0::,0:cols-1]
	subset_features.append(temp_features)


clf = []
for i in range(10):
	temp = tree.DecisionTreeClassifier()
	temp.fit(temp_features,temp_label)
	clf.append(temp)

	




test_label = test_data['label']
test_label = test_label.as_matrix()
test_data = test_data.as_matrix()
test_features = test_data[0::,0:cols-1]



for label in training:
	if(labels=='sumuf.' or labels=='land.' or labels=='neptune.' or labels=='pod.' or labels=='teardrop.' ):
		labels = 'dos'

	if(labels=='buffer_overflow.' or labels =='loadmodule.' or labels=='perl.' or labels =='rootkit.' ):
		labels = 'u2r'

	if(labels=='ftp_write.' or labels=='gues_passwd.' or labels=='imap.' or labels=='multihop.' or labels =='phf.' or labels=='warezclient.' or labels=='warezmaster'):
		labels = 'r2l'

	if(labels=='ipsweep.' or labels=='nmap.' or labels == 'portsweep.' or labels=='statn.'):
		label = 'probe'

#print(training['label'].value_counts())



unit = 494021/10
subset1_features = training_features[0:unit]
subset1_label = training_label[0:unit]

subset2_features = training_features[unit:2*unit]
subset2_label = training_label[unit:2*unit]

subset3_features = training_features[2*unit:3*unit]
subset3_label = training_label[2*unit:unit]
mnb = MultinomialNB()
pred2 = mnb.fit(pred1,testing_label).predict(pred1)
acc = accuracy_score(testing_label,pred2)
print(acc)
gnb = GaussianNB()
gnb.fit(table[0::,0::],training_label)
pred2 = gnb.predict()
acc = accuracy_score()
print(acc)



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




training_label.astype(np.float)
print(training_label.shape)
print(training_label)


gnb = GaussianNB()
gnb.fit(table,training_label)









