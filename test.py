# importing necessary packages
import numpy as np
import pandas as pd
import os
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import svm
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
torch.manual_seed(0)


# defining paths
path_to_train = "./sign_data/sign_data/train/"
path_to_test = "./sign_data/sign_data/test/"
train_classes = os.listdir(path_to_train)
test_classes = os.listdir(path_to_test)
# defining y sets for test and train
y_train = []
y_test = []

#test - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# calculating number of test images 
value_of_img1 = 0
for ii in test_classes:
    if os.path.isdir(path_to_test+ii) == 1:
        value_of_img1 += len(os.listdir(path_to_test+ii))

# creating dataset for test
datasetTest = np.ndarray(shape=(500,83,229,3),
                     dtype=np.float32)

# getting images, transfroming them to numpy
pointer1 = 0
for i in test_classes:
    if os.path.isdir(path_to_test+i):
        imgs = os.listdir(path_to_test+i)
    for j in imgs:
        if os.path.isdir(path_to_test+i) == 1 and j.find("DS") == -1:
            imgg = load_img(path_to_test + i + "/" + j)  # this is a PIL image
            img = imgg.resize((229, 83))
            # Convert to Numpy Array
            x = img_to_array(img)  
            x = x.reshape((83,229,3))
            # Normalize
            x = (x - 128.0) / 128.0
            datasetTest[pointer1] = x
            pointer1 += 1
            if i.find('forg') != -1:
                y_test.append(0)
            else:
                y_test.append(1)

#train - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# calculating number of train images 
value_of_img = 0
for ii in train_classes:
    value_of_img += len(os.listdir(path_to_train+ii))

# creating dataset for train
dataset = np.ndarray(shape=(value_of_img,83,229,3),
                     dtype=np.float32)

# getting images, transfroming them to numpy
pointer = 0
for i in train_classes:
    imgs = os.listdir(path_to_train+i)
    for j in imgs:
        imgg = load_img(path_to_train + i + "/" + j)  # this is a PIL image
        img = imgg.resize((229, 83))
        # Convert to Numpy Array
        x = img_to_array(img)  
        x = x.reshape((83,229,3))
        # Normalize
        x = (x - 128.0) / 128.0
        dataset[pointer] = x
        pointer += 1
        if i.find('forg') != -1:
            y_train.append(0)
        else:
            y_train.append(1)

# reshaping arrays from 4d to 2d 
X = dataset.reshape((1649,83*229*3))
X_test = datasetTest.reshape((500,83*229*3))
y = y_train

#training
clf = svm.SVC()
clf.fit(X, y)

#predicting
clf.predict(X_test)
print(clf.score(X_test, y_test))

