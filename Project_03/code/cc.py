"""
    UB_ID : 50291708
    Name  : Md Moniruzzaman Monir
    
"""

from vj import VJ
import pickle


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


class CC():
    def __init__(self, layers):
        self.layers = layers
        self.clfs = []

    def train(self, training):
        pos, neg = [], []
        for ex in training:
            if ex[1] == 1:
                pos.append(ex)
            else:
                neg.append(ex)

        for feature_num in self.layers:
            if len(neg) == 0:
                break
            clf = VJ(feature_num)
            clf.train(pos+neg, len(pos), len(neg))
            self.clfs.append(clf)
            false_positives = []
            for ex in neg:
                if self.classify(ex[0]) == 1:
                    false_positives.append(ex)
            neg = false_positives

    def classify(self, image, scale, x, y):
        for clf in self.clfs:
            if clf.classify(image, scale, x, y) == 0:
                return 0
        return 1

    def scaleFeatures(self, scale):
         for clf in self.clfs:
             clf.scaleFeatures(scale)

    def save(self, filename):
        with open(filename+".pkl", 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename+".pkl", 'rb') as f:
            return pickle.load(f)
