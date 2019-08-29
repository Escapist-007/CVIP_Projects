"""
    UB_ID : 50291708
    Name  : Md Moniruzzaman Monir

"""

import cv2
import glob
import numpy as np
import pickle
import time

faceDir_train     = './dataset/train/face/*.pgm'
nonfaceDir_train  = './dataset/train/non-face/*.pgm'

# 0 --> grayscale
face_images       = [(cv2.imread(file,0),1) for file in glob.glob(faceDir_train)]
nonface_images    = [(cv2.imread(file,0),0) for file in glob.glob(nonfaceDir_train)]

face_count    = len(face_images)
nonface_count = len(nonface_images)

train_ds = face_images + nonface_images

# info about dataset
print("Total face image: " + str(face_count))
print("Total non-face image: " + str(nonface_count))
print("Total image: " + str(len(train_ds)))

with open("train_ds.pkl", 'wb') as f:
    pickle.dump(train_ds,f)


def calcIntegral(img):
    """
        This method returns the integral image of a given image
        ------
        | Args:|
        ------
        img: A 2d-numpy array of original image
        
        """
    rows = img.shape[0]
    cols = img.shape[1]
    
    new_img = np.zeros((rows,cols))
    
    new_img[0][0] = img[0][0]
    
    '''
        1st row calculation
        '''
    for c in range(1,cols):
        new_img[0][c] = new_img[0][c-1] + img[0][c]
    
    '''
        1st column calculation
        '''
    for r in range(1,rows):
        new_img[r][0] = new_img[r-1][0] + img[r][0]
    
    '''
        Other cell calculation
        '''
    for r in range(1,rows):
        for c in range(1,cols):
            new_img[c][r] = (new_img[c-1][r]+new_img[c][r-1]-new_img[c-1][r-1]) + (img[c][r])
    
    return new_img


train_ds_integral  = []

for x in range(len(train_ds)):
    integral_img = calcIntegral(train_ds[x][0])
    label        = train_ds[x][1]
    train_ds_integral.append((integral_img,label))


# saving
with open("train_integral_ds.pkl", 'wb') as f:
    pickle.dump(train_ds_integral,f)
