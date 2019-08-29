"""
    UB_ID : 50291708
    Name  : Md Moniruzzaman Monir
    
"""

import numpy as np
import pickle
import time
from rectangleArea import Rectangle
from feature import Feature


with open("train_ds.pkl", 'rb') as f:
    train_ds = pickle.load(f)

with open("train_integral_ds.pkl", 'rb') as f:
    train_ds_integral = pickle.load(f)


# creating all features for a certain window shape
s = time.time()

f = Feature(train_ds[0][0].shape)
features = f.creating_all_features()

e = time.time()

print("Feature creation is done")
print("Time for creating all features :")
print(e-s)


start = time.time()

X1, y1 = f.features_value(train_ds_integral)

end = time.time()

print("Feature value is calculated for all training images")
print("Time:")
print(end - start)


total_features = len(X1)

X_first_half   = X1[:total_features//2,]
X_second_half  = X1[ total_features//2:,]

print("Saving the features' value for training iamges")
with open("features_value_1.pkl", 'wb') as f:
      pickle.dump(X_first_half,f)

with open("features_value_2.pkl", 'wb') as f:
      pickle.dump(X_second_half,f)


with open("features_value_1.pkl", 'rb') as f:
      a = pickle.load(f)

with open("features_value_2.pkl", 'rb') as f:
      b = pickle.load(f)

X2 = np.concatenate((a,b), axis=0)

with open("y.pkl", 'wb') as f:
      pickle.dump(y1,f)
