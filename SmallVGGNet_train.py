# -*- coding: utf-8 -*-
"""
Created on Sat May 16 16:08:07 2020

@author: AMAZING-GRACE
"""

import matplotlib
matplotlib.use("Agg")

import SmallVGGNet
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True, 
                help = "path to input dataset of images")
ap.add_argument("-m", "--model", required = True, 
                help = "path to output trained model")
ap.add_argument("-l", "--label-bin", required = True, 
                help = "path to output label binarizer")
ap.add_argument("-p", "--plot", required = True, 
                help = "path to output accuracy/loss plot")
args = vars(ap.parse_args())

print("[INFO] loading images...")
data = []
labels = []

imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (64, 64))
    data.append(image)
    
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)
    
data = np.array(data, dtype = "float") / 255.0
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state = 42)

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, 
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")

model = SmallVGGNet.build(width=64, height=64, depth=3, classes=len(lb.classes_))

INIT_LR = 0.01
EPOCHS = 75
batch_size = 32


print("[INFO] training network...")
opt = SGD(lr=INIT_LR, decay=INIT_LR/EPOCHS)
model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics = ["accuracy"])

H = model.fit(x = aug.flow(trainX, trainY, batch_size = batch_size), 
              validation_data = (testX, testY), 
              steps_per_epoch = len(trainX) // batch_size, epoch = EPOCHS)

print("[INFO] evaluating network...")
predictions = model.predict(x=testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1),
                            target_names=lb.classes_))

N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["train_acc"], label="train_acc")
plt.plot(N, H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy (SmallVGGNet)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["plot"])

print("[INFO] serialising network and label binarizer...")
model.save(args["model"], save_format="h5")
f = open(args["label_bin"], 'wb')
f.write(pickle.dumps(lb))
f.close()
