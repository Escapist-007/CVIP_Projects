"""
    UB_ID : 50291708
    Name  : Md Moniruzzaman Monir
    
"""
import numpy as np
import pickle
import time
import math
from feature import Feature
from rectangleArea import Rectangle
from weakClassifier import wc
from integral import calcIntegral

class VJ:
    # constructor
    def __init__(self, total_wc=100):
        
        self.total_wc           = total_wc
        self.classifiers        = []
        self.classifiersWeights = []
    
    # train the model
    def train(self, train_ds):
        
        face_count = 0
        nonface_count = 0
        image_count = len(train_ds)
        
        # face and nonface image count
        for x in range(image_count):
            label = train_ds[x][1]
            if label==1:  # Face
                face_count = face_count + 1
            else:
                nonface_count = nonface_count + 1
        
        # loading integral images of train_ds
        with open("train_integral_ds.pkl", 'rb') as f:
            train_ds_integral = pickle.load(f)
            print("Integral images are loaded from pickle file for training the model")
        

        w = np.zeros(image_count)  # sample_Weight
        
        for x in range(image_count):
            # Initial weight of every image (sample weight)
            if label == 1:  # Face
                w[x] = 1.0 / (2*face_count)
            else:
                w[x] = 1.0 / (2*nonface_count)
        
        # feature generation
        f = Feature(train_ds[0][0].shape)
        features = f.creating_all_features()
        
        # load features value (X) and classification (y) from pickle file
        # saves a lot of time while tuning in training phase
        with open("features_value_1.pkl", 'rb') as f:
            a = pickle.load(f)
        with open("features_value_2.pkl", 'rb') as f:
            b = pickle.load(f)
            X = np.concatenate((a,b), axis=0)
        with open("y.pkl", 'rb') as f:
            y = pickle.load(f)
                
        # Selecting weak classifiers
        for wc_i in range(self.total_wc):
            
            w = w / np.linalg.norm(w)
            
            trainedWeakClassifier = self.weakClassifier_training(X, y, features, w, face_count,nonface_count)
            
            bc = None
            bacc = None
            be = float('inf')
            
            for twc in trainedWeakClassifier:
                e = 0
                acc = []
                        
                for integral_image, ww in zip(train_ds_integral,w):
                   
                    predicted_label = twc.classify(integral_image[0])
                    true_label = integral_image[1]
                    acc.append(abs(predicted_label-true_label))
                    
                    e += ww * abs(predicted_label-true_label)
                
                e = e / len(train_ds_integral)
                
                if e < be:
                    bc = twc
                    be = e
                    bacc = acc
            
            beta = be / (1.0 - be)
            
            for i in range(len(bacc)):
                w[i] = w[i] * (beta ** (1 - bacc[i]))
            
            alpha = math.log(1.0/beta)
            self.classifiersWeights.append(alpha)
            self.classifiers.append(bc)

    def weakClassifier_training(self, X, y, features, weights, face_count,nonface_count):
        total_pos_weights = 0
        total_neg_weights = 0
        for w, label in zip(weights, y):
            if label == 1:
                total_pos_weights = total_pos_weights + w
            else:
                total_neg_weights = total_neg_weights + w
        c = []
        
        for index, feature in enumerate(X):
            # sort according to feature value
            sortedList = sorted(zip(weights,feature, y), key=lambda x: x[1])
            pos_img_seen = 0
            neg_img_seen = 0
            pos_img_weights = 0
            neg_img_weights = 0
            min_e = float('inf')
            bf = None
            bt = None
            bp = None
            for w, f, label in sortedList:
                error = min(neg_img_weights + total_pos_weights - pos_img_weights, pos_img_weights + total_neg_weights - neg_img_weights)
                
                if error < min_e:
                    min_e = error
                    bf = features[index]
                    bt = f
                    if pos_img_seen > neg_img_seen:
                        bp = 1
                    else:
                        bp = -1
                if label == 1:
                    pos_img_seen += 1
                    pos_img_weights += w
                else:
                    neg_img_seen += 1
                    neg_img_weights += w
                        
            black_region = bf[0]
            white_region = bf[1]

            weak_c = wc(black_region,white_region,bt,bp)
            c.append(weak_c)
        return c

    def classify(self, image, s, x, y):
        total = 0
        integral_image = calcIntegral(image)
        
        for c_w, cl in zip(self.classifiersWeights, self.classifiers):
            total += c_w * cl.classify(integral_image,s,x,y)
        
        if total >= 0.5 * sum(self.classifiersWeights):
            return 1
        else:
            return 0

    def saveModel(self, filename):
        with open(filename+".pkl", 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def loadModel(filename):
        with open(filename+".pkl", 'rb') as f:
            return pickle.load(f)
