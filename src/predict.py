
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
import lib_IO


training_samples = 278
test_samples = 138
x_shape = 176
y_shape = 208
z_shape = 176

final = x_shape*y_shape*z_shape
sample_size = 1000
samples = np.zeros((sample_size,3))

"""for i in range(0,sample_size):
samples[i,0]=random.randrange(0, 176)
samples[i,1]=random.randrange(0, 208)
samples[i,2]=random.randrange(0, 176)"""

train = np.zeros((training_samples,final))
test = np.zeros((test_samples,final))
train_labels = pd.read_csv('targets.csv', header=None)
counter = 0
cols = []

print("Traversing images and shaping to feature vector")
for file in os.listdir('data/set_train'):
    print(counter)
    file_path = os.path.join('data/set_train', file)
    img = nib.load(file_path)
    img_data = img.get_data()
    img_data = img_data[:,:,:,0]
    train[counter,:] = img_data.reshape(final)
    counter += 1

mask = (train < 10)
idx = mask.any(axis=0)

counter = 0
for file in os.listdir('data/set_test'):
print(counter)
file_path = os.path.join('data/set_test', file)
img = nib.load(file_path)
img_data = img.get_data()
img_data = img_data[:,:,:,0]
test[counter,:] = img_data.reshape(final)
counter += 1

train = train[:,~idx]
test = test[:,~idx]
print(train.shape)
print(train)
print(test.shape)
print(test)

#train = train[:,:10]
#test = test[:,:10]

print("Dimensionality Reduction")
"""pca = PCA(150, 'randomized',
          whiten=True).fit(train)
train_pca = pca.transform(train)
test_pca = pca.transform(test)
print(train_pca.shape)"""

"""print("Train Linear SVC")
model = LinearSVC()
model.fit(train,train_labels.values.ravel())

print("Begin predictions")
print(model.predict(test))"""

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(train, train_labels.values.ravel())

print(regr.predict(test))
print("Next")


model = LinearSVC()
model.fit(train,train_labels.values.ravel())
y_predict = model.predict(test)
print("Done")

write_Y("prediction.csv",y_predict,np.arange(start=1,stop=138,dtype=np.uint32))

# returns numpy array of data. Dimensions: samplesCount X 6
# features are [abs CSF volume,  abs GM volume, abs WM volume, rel CSF volume, rel GM volume, rel WM volume]
#	samples count:
# 	training: 278
# 	testing: 138

def parseDataSetXML(samplesCount,filepath)
	files = [f for f in os.listdir(filepath)]
	features_matrix = np.zeros((samplesCount,6))

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

		cAbs = vol_abs_CGW.split("[")[1].split(" ")[0]
		gAbs = vol_abs_CGW.split("[")[1].split(" ")[1]
		wAbs = vol_abs_CGW.split("[")[1].split(" ")[2]
		cRel = vol_rel_CGW.split("[")[1].split(" ")[0]
		gRel = vol_rel_CGW.split("[")[1].split(" ")[1]
		wRel = vol_rel_CGW.split("[")[1].split(" ")[2]

		features_matrix[int(index)-1,:] = np.array([float(cAbs),float(gAbs),float(wAbs),float(cRel),float(gRel),float(wRel)])
	return features_matrix

def write_Y(fname, Y_pred, Ids):
	if Y_pred.shape[0] != Ids.shape[0]:
		print("error Ids- dimension of y matrix does not match number of expected predictions")
		print('y: {0} - expected: {1}'.format(Y_pred.shape,Ids.shape))
	else:
		f = open(fname, 'w+')
		np.savetxt(fname=f,X= np.column_stack([Ids,Y_pred]),
				   fmt=['%d', '%d'],delimiter=',',header='Id,Prediction',comments='')




