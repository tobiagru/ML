
#!/usr/bin/env python
import os
import numpy as np
import nibabel as nib
import sklearn as skl
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
import random
from sklearn import datasets, linear_model
from xml.dom import minidom
from sklearn import svm, linear_model, preprocessing
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFE


def parsecatRoi(samplesCount,filepath):


	files = [f for f in os.listdir(filepath)]
	features_matrix = np.empty((samplesCount,544))
	counter =0
	for f in files:
		# print(index)

		if(f==".DS_Store"):
			continue
		xmldoc = minidom.parse(filepath + '/'+f)

		if(xmldoc.getElementsByTagName('Vcsf')):
			Vcsf1 = (xmldoc.getElementsByTagName('Vcsf')[0].childNodes[0].data)
		else:
			print("Attributes missing at index ")

		if(xmldoc.getElementsByTagName('Vcsf')):
			Vcsf2 = (xmldoc.getElementsByTagName('Vcsf')[1].childNodes[0].data)
		else:
			print("Attributes missing at index ")

		if(xmldoc.getElementsByTagName('Vgm')):
			Vgm1 = (xmldoc.getElementsByTagName('Vgm')[0].childNodes[0].data)
		else:
			print("Attributes missing at index ")

		if(xmldoc.getElementsByTagName('Vgm')):
			Vgm2 = (xmldoc.getElementsByTagName('Vgm')[1].childNodes[0].data)
		else:
			print("Attributes missing at index ")
		if(xmldoc.getElementsByTagName('Vgm')):
			Vgm3 = (xmldoc.getElementsByTagName('Vgm')[2].childNodes[0].data)
		else:
			print("Attributes missing at index ")

		if(xmldoc.getElementsByTagName('Vwm')):
			Vwm1 = (xmldoc.getElementsByTagName('Vwm')[0].childNodes[0].data)
		else:
			print("Attributes missing at index ")

		features = np.empty((1, 0))
		Vcsf1 = Vcsf1.split("[")[1].split("]")[0].split(";")
		Vcsf2 = Vcsf2.split("[")[1].split("]")[0].split(";")
		Vgm1 = Vgm1.split("[")[1].split("]")[0].split(";")
		Vgm2 = Vgm2.split("[")[1].split("]")[0].split(";")
		Vgm3 = Vgm3.split("[")[1].split("]")[0].split(";")
		Vwm1 = Vwm1.split("[")[1].split("]")[0].split(";")
		total_size = len(Vcsf1) + len(Vcsf2) +len( Vgm1) +len( Vgm2) +len(Vgm3)+len(Vwm1)
		for i in Vcsf1:
			features = np.append(features, float(i))
		for i in Vcsf2:
			features = np.append(features, float(i))
		for i in Vgm1:
			features = np.append(features, float(i))
		for i in Vgm2:
			features = np.append(features, float(i))
		for i in Vgm3:
			features = np.append(features, float(i))
		for i in Vwm1:
			features = np.append(features, float(i))
		features_matrix[counter] = features
		counter =counter +1
	return features_matrix



def parseReport(samplesCount,filepath):
	files = [f for f in os.listdir(filepath)]
	features_matrix = np.zeros((samplesCount,7))

	for f in files:
		# print(f)
		if(f==".DS_Store"):
			continue
	
		xmldoc = minidom.parse(filepath + '/'+f)
		filenmae = xmldoc.getElementsByTagName('file')[0].childNodes[0].data
		index = filenmae.split("_")[1]
		# print(index)
		values = xmldoc.getElementsByTagName('file')[0].childNodes[0].data

		if(xmldoc.getElementsByTagName('vol_abs_CGW')):
			vol_abs_CGW = xmldoc.getElementsByTagName('vol_abs_CGW')[0].childNodes[0].data
		else:
			print("Attributes missing at index " + index)
		if(xmldoc.getElementsByTagName('vol_rel_CGW')):
			vol_rel_CGW = xmldoc.getElementsByTagName('vol_rel_CGW')[0].childNodes[0].data
		else:
			print("Attributes missing at index " + index)
		if(xmldoc.getElementsByTagName('vol_TIV')):
			vol_TIV = (xmldoc.getElementsByTagName('vol_TIV')[0].childNodes[0].data)
		else:
			print("Attributes missing at index " + index)

		cAbs = vol_abs_CGW.split("[")[1].split(" ")[0]
		gAbs = vol_abs_CGW.split("[")[1].split(" ")[1]
		wAbs = vol_abs_CGW.split("[")[1].split(" ")[2]
		cRel = vol_rel_CGW.split("[")[1].split(" ")[0]
		gRel = vol_rel_CGW.split("[")[1].split(" ")[1]
		wRel = vol_rel_CGW.split("[")[1].split(" ")[2]

		features_matrix[int(index)-1,:] = np.array([float(cAbs),float(gAbs),float(wAbs),float(cRel),float(gRel),float(wRel),float(vol_TIV)])
	return features_matrix



training_samples = 278
test_samples = 138

train2= parsecatRoi(training_samples, "./data/train/label")

print(train2.shape)



train = parseReport(training_samples, "./data/train/report")

print(train.shape)




train = np.append(train, train2, axis=1 )
print(train.shape)



test = parseReport(test_samples, './data/test/report')
print(test.shape)
test2=parsecatRoi(test_samples, './data/test/label')
print(test2.shape)

test = np.append(test, test2,axis=1)

print(test.shape)


train_labels = pd.read_csv('./data/train/targets.csv', header=None)
labels = np.array(train_labels)
print("shape is " +str( train.shape))




print("Model Selection")
regr = linear_model.LogisticRegression(C=0.1,penalty='l1', tol=0.01)
sfm = SelectFromModel(regr)
sfm.fit(train,labels)	

train = sfm.transform(train)
test = sfm.transform(test)

print("Regression")
regr.fit(train,labels)
res = regr.predict_proba(test)
cl = regr.predict(test)
print("Classes")
print(cl)

result = np.zeros((test_samples,2))

test_sequence = np.arange(1,test_samples+1)

print("Writing Result")
for i in range(0,test_samples):
 result[i,0] = (test_sequence[i]).astype(int)
 print(res[i,0],res[i,1])
 result[i,1] = (res[i,1]).astype(float)

print(result)
 
np.savetxt("result_preProcessed.csv",result,fmt=['%f','$f'],delimiter=",",header="ID,Prediction",comments="")