"""
    UB_ID : 50291708
    Name  : Md Moniruzzaman Monir
    
"""

class wc:
    # constructor
    def __init__(self, black, white, threshold, polarity):
        self.black = black
        self.white = white
        self.threshold = threshold
        self.polarity = polarity

    def classify(self, integral_img, s, x, y):

        black_value = 0.0
        white_value = 0.0

        for br in self.black:
            black_value += br.compute_sum(integral_img, s, x, y)
        for wr in self.white:
            white_value += wr.compute_sum(integral_img, s, x, y)

        value = int(black_value - white_value)

        if self.polarity*value < self.polarity*self.threshold:
            return 1
        else:
            return 0

