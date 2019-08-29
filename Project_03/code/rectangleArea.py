"""
    UB_ID : 50291708
    Name  : Md Moniruzzaman Monir
    
"""

class Rectangle:
    # constructor
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    
    # Return the sum of all pixels inside a rectangle for a specific integral image
    def compute_sum(self, integralImg, scale, x, y):
        
        x = self.x
        y = self.y
        width  = self.width
        height = self.height
        
        one   = integralImg[y][x]
        two   = integralImg[y][x+width]
        three = integralImg[y+height][x]
        four  = integralImg[y+height][x+width]
        
        desiredSum = (one + four) - (two + three)
        
        return desiredSum
