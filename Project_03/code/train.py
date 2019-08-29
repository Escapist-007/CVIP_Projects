"""
    UB_ID : 50291708
    Name  : Md Moniruzzaman Monir
    
"""
import cv2
import os
import sys
import glob
import numpy as np
import random as rand
import matplotlib.pyplot as plt
import pickle
import time
from vj import VJ
from cc import CC

faceDir_train     = './dataset/train/face/*.pgm'
nonfaceDir_train  = './dataset/train/non-face/*.pgm'

# 0 --> grayscale
face_images       = [(cv2.imread(file,0),1) for file in glob.glob(faceDir_train)]
nonface_images    = [(cv2.imread(file,0),0) for file in glob.glob(nonfaceDir_train)]

face_count    = len(face_images)
nonface_count = len(nonface_images)

train_ds = face_images + nonface_images

print("Total face image: " + str(face_count))
print("Total non-face image: " + str(nonface_count))
print("Total image: " + str(len(train_ds)))

with open("train_ds.pkl", 'wb') as f:
    pickle.dump(train_ds,f)

layers = [1, 2, 5, 10, 50]
cc = CC(layers)
cc.train(train_ds)
filename = 'trained_model'
cc.save(filename)
