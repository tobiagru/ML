"""
http://www.sciencedirect.com/science/article/pii/S1053811910000108

regression according to above paper
"""


import os
import numpy as np
import pandas as pd
import nibabel as nib
import nilearn.image as nimg
from sklearn.decomposition import PCA
from sklearn.linear_model import ARDRegression
from skrvm import RVR
from xml.dom import minidom


#data specs
training_samples = 278
test_samples = 138
x_shape = 176
y_shape = 208
z_shape = 176

train = np.zeros((278, 6443008), dtype=np.uint16)

#Loading & Preprocessing
dir = 'data/train/mri'
targets = pd.read_csv("data/targets.csv", header=None)
for file in os.listdir(dir):
    _, file_ext = os.path.splitext(file)
    if file_ext == ".nii" and file[0:3] == "mwp1":
        #create full path to file
        file_path = os.path.join(dir, file)

        #get number of the image
        file_num = [int(s) for s in file.split() if s.isdigit()]
        file_num = file_num[-1]

        #load .nii image
        img = nib.load(file_path)
        print ("found file num {0} with {1} shape".format(file_num, img.shape))

        #Affine Registration
        img = nimg.reorder_img(img)

        #Kernel Smoothing
        #   FWHM Kernel
        #   8mm smoothing
        img = nimg.smooth_img(img, 8)

        #get data
        train_tmp = img.get_data()

        #flatten data
        train_tmp.flatten()

        #add to train array
        train[file_num,:] = train_tmp







"""
#PCA
#   410 components
pca = PCA(270).fit(train)
print(pca.explained_variance_ratio_)
train_pca = pca.transform(train)



#Relevance Vector Regression
#   zero mean Gaussian prior
#   poly1 regression
rvr = RVR(kernel="rbf")
rvr.train(train_pca, targets)

#Loading & Preprocessing
dir = 'data/test/mri'
targets = pd.read_csv("data/targets.csv", header=None)
for file in os.listdir(dir):
    _, file_ext = os.path.splitext(file)
    if file_ext == ".nii" and file[0:3] == "mwp1":
        #create full path to file
        file_path = os.path.join(dir, file)

        #load .nii image
        img = nib.load(file_path)


        #Affine Registration
        img = nimg.reorder_img(img)

        #Kernel Smoothing
        #   FWHM Kernel
        #   8mm smoothing
        img = nimg.smooth_img(img, 8)

#pca on test data
test_pca = pca.transform(test)


#Predict
predicts = rvr.predict(test_pca)

#write to file


"""