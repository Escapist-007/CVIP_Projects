"""
    UB_ID : 50291708
    Name  : Md Moniruzzaman Monir
    
"""


import numpy as np
import pickle
import time
from rectangleArea import Rectangle

class Feature:
    # constructor
    def __init__(self, image_shape):
        
        self.height, self.width = image_shape
        self.f = None  # Feature list
        self.f_values = None # Features' values for all images

    def creating_all_features(self):
        
        '''
            Create 5 types of Haar Features for all sizes, shapes and positions in a fixed window
            '''
        height = self.height
        width  = self.width
        
        # List of tuple where 1st element means List of black rectangles and 2nd element means List of white rectangles
        features = []
        
        for w in range(1, width+1):      # All possible width
            for h in range(1, height+1): # All possible height
                
                i = 0
                while i + w < width:
                    j = 0
                    while j + h < height:
                        
                        fixed   = Rectangle(i, j, w, h)
                        right_1 = Rectangle(i+1*w, j, w, h)
                        right_2 = Rectangle(i+2*w, j, w, h)
                        
                        bottom_1_right_1 = Rectangle(i+1*w, j+1*h, w, h)
                        
                        bottom_1 = Rectangle(i, j+1*h, w, h)
                        bottom_2 = Rectangle(i, j+2*h, w, h)
                        
                        '''
                            2 Rectangle Haar Features
                            '''
                        # Horizontal  -->  fixed (white) | right_1 (black)
                        if i + 2 * w < width:
                            features.append(([right_1], [fixed]))
                        
                        # Vertical -->  fixed(black)
                        #  ------------
                        #   bottom_1(white)
                        if j + 2 * h < height:
                            features.append(([fixed], [bottom_1]))
                        
                        
                        
                        '''
                            3 Rectangle Haar Features
                            '''
                        # Horizontal -->  fixed (white) | right_1 (black) | right_2 (white)
                        if i + 3 * w < width:
                            features.append(([right_1], [right_2, fixed]))
                        
                        # Vertical -->  fixed(white)
                        #  ------------
                        #   bottom_1(black)
                        #  ------------
                        #   bottom_2(white)
                        if j + 3 * h < height:
                            features.append(([bottom_1], [bottom_2, fixed]))
                        
                        
                        '''
                            4 Rectangle Haar Features
                            '''
                        
                        if i + 2 * w < width and j + 2 * h < height:
                            features.append(([right_1, bottom_1], [fixed, bottom_1_right_1]))
                        
                        j += 1
                    i += 1
        
        features = np.array(features)
        self.f = features
        return features

    def features_value(self, train_ds_integral):
        '''
            Save features' value across all training images
            '''
        
        X = np.zeros((len(self.f), len(train_ds_integral)))
        y = np.array(list(map(lambda data: data[1], train_ds_integral)))
        
        feature_idx = 0
        
        for black_regions, white_regions in self.f:
            for k in range(len(train_ds_integral)):
                
                integral_img = train_ds_integral[k][0]
                black_value = 0
                white_value = 0
                
                for br in black_regions:
                    black_value += br.compute_sum(integral_img)
                for wr in white_regions:
                    white_value += wr.compute_sum(integral_img)
                
                X[feature_idx][k] = (black_value - white_value)
            
            feature_idx += 1
        
        self.f_values = (X,y)
        return X, y
