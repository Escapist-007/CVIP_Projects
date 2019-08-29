"""
    UB_ID : 50291708
    Name  : Md Moniruzzaman Monir
    
"""
import numpy as np
import pickle
import time

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
