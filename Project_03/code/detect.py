import sys
import numpy as np
import cv2
import pickle
import json
from vj import VJ
from cc import CC
import glob

json_list = []

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

# Loading the test images
face_images     = []

directory = sys.argv[1]
imageDir  = directory + '/*.jpg'
face_images    = [cv2.imread(file,0) for file in glob.glob(imageDir)]

count = len(face_images)
with open("test.pkl", 'rb') as f:
    json_lists = pickle.load(f)

print(" Finished loading the images and the trained viola jones model ")

def zero_pad(img, pwx, pwy):
    """Pads a given image with zero at the border."""
    padded_img = copy.deepcopy(img)
    for i in range(pwx):
        padded_img.insert(0, [0 for value in enumerate(padded_img[i])])
        padded_img.insert(len(padded_img), [0 for value in enumerate(padded_img[-1])])
    for i, row in enumerate(padded_img):
        for j in range(pwy):
            row.insert(0, 0)
            row.insert(len(row), 0)
    return padded_img

def crop(img, xmin, xmax, ymin, ymax):
    """Crops a given image."""
    if len(img) < xmax:
        print('WARNING')
    patch = img[xmin: xmax]
    patch = [row[ymin: ymax] for row in patch]
    return patch


for i in range(count):
    print("Detecting face in image : " + str(i) + ".jpg")
    img = face_images[i]
    h, w = img.shape
    rects = []
    scale  = 1
    size   = 19
    scaling_factor = 1.5

    while size <= min(h, w):
        y = 0
        while y + size <= h:
            if size < 100:
                break
            x = 0
            while x + size <= w:
                rects.append([x, y, size, size])
                x += 10
            y += 10

        scale *= scaling_factor
        size = int(size * scaling_factor)

        if size >= 712:
            break

    # Non maximum suppression
    rects = list(cv2.groupRectangles(rects, 1, 0.5)[0])

    index = i + 1
    img_name = str(index)+'.jpg'
    l = np.array(rects).tolist()

    for box in l:
        element = {"iname": img_name, "bbox": box}
        json_list.append(element)


output_json = "results.json"
with open(output_json, 'w') as f:
    json.dump(json_lists, f)
