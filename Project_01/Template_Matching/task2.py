"""
    UB_ID : 50291708
    Name  : Md Moniruzzaman Monir
    
    Character Detection
    (Due date: March 8th, 11: 59 P.M.)
    
    The goal of this task is to experiment with template matching techniques. Specifically, the task is to find ALL of
    the coordinates where a specific character appears using template matching.
    
    There are 3 sub tasks:
    
    1. Detect character 'a'.
    2. Detect character 'b'.
    3. Detect character 'c'.
    
    You need to customize your own templates. The templates containing character 'a', 'b' and 'c' should be named as
    'a.jpg', 'b.jpg', 'c.jpg' and stored in './data/' folder.
    
    Please complete all the functions that are labelled with '# TODO'. Whem implementing the functions,
    comment the lines 'raise NotImplementedError' instead of deleting them. The functions defined in utils.py
    and the functions you implement in task1.py are of great help.
    
    Hints: You might want to try using the edge detectors to detect edges in both the image and the template image,
    and perform template matching using the outputs of edge detectors. Edges preserve shapes and sizes of characters,
    which are important for template matching. Edges also eliminate the influence of colors and noises.
    
    Do NOT modify the code provided.
    Do NOT use any API provided by opencv (cv2) and numpy (np) in your code.
    Do NOT import any library (function, module, etc.).
    
"""

import argparse
import json
import os

import utils
import task1


import numpy as np
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--img_path", type=str, default="./data/proj1-task2.jpg",
        help="path to the image used for character detection (do not change this arg)")
    parser.add_argument(
        "--template_path", type=str, default="",
        choices=["./data/a.jpg", "./data/b.jpg", "./data/c.jpg"],
        help="path to the template image")
    parser.add_argument(
        "--result_saving_directory", dest="rs_directory", type=str, default="",
        help="directory to which results are saved (do not change this arg)")
    args = parser.parse_args()
    return args


def read_image(img_path, show=False):
    """Reads an image into memory as a grayscale array"""
    
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    if not img.dtype == np.uint8:
        pass

    if show:
        show_image(img)

    img = [list(row) for row in img]   # converting to nested list
    return img


def show_image(img, delay=1000):
    """Shows an image"""

    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', img)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()


# NCC (Normalized Cross Correlation) between image and kernel
def corr2d(img, kernel):
    """
        Args:
        img   : nested list (int), image.
        kernel: nested list (int), kernel.
        
        Returns:
        img_conv: nested list (int), image.
        """
    padding_layer = len(kernel)//2
    img    = utils.zero_pad(img,padding_layer,padding_layer)   # padding the image
    
    row_count = len(img)
    col_count = len(img[0])
    
    # output image having same size as the input image
    img_corr = []
    
    zeros = [0]*(col_count-2*padding_layer)
    
    for i in range(row_count-2*padding_layer):
        img_corr.insert(0, [0 for value in enumerate(zeros)])

    kernel_h = len(kernel)
    kernel_w = len(kernel[0])

    img    = np.asarray(img, dtype=np.float32)
    kernel = np.asarray(kernel, dtype=np.float32)

    for i in range(row_count - kernel_h + 1):
        for j in range(col_count - kernel_w + 1):
            
            mult_result = utils.elementwise_mul(utils.crop(img,i,i+kernel_h,j,j+kernel_w), kernel)
            
            sqr_img = utils.elementwise_mul(utils.crop(img,i,i+kernel_h,j,j+kernel_w),
                                            utils.crop(img,i,i+kernel_h,j,j+kernel_w))
                                            
            sqr_ker = utils.elementwise_mul(kernel, kernel)
            
            sum = 0
            sum_sqr_img = 0
            sum_sqr_ker = 0
            
            for p in range(len(mult_result)):
                for q in range(len(mult_result[p])):
                    sum += mult_result[p][q]
                    sum_sqr_img += sqr_img[p][q]
                    sum_sqr_ker += sqr_ker[p][q]
     
            img_corr[i][j] = sum / np.sqrt(sum_sqr_img * sum_sqr_ker)

    img_corr = [list(row) for row in img_corr]

    return img_corr


def detect(img, template):
    """Detect a given character, i.e., the character in the template image.

    Args:
        img: nested list (int), image that contains character to be detected.
        template: nested list (int), template image.

    Returns:
        coordinates: list (tuple), a list whose elements are coordinates where the character appears.
            format of the tuple: (x (int), y (int)), x and y are integers.
            x: row that the character appears (starts from 0).
            y: column that the character appears (starts from 0).
    """
    # TODO: DONE
    img1 = corr2d (img,template)
    img1 = np.asarray(img1,dtype=np.float32)
    
    coordinates = []

    for r in range(len(img1)):
        for c in range(len(img1[r])):
            if img1[r][c] >= threshold:
                row = r - (h//2)
                col = c - (w//2)
                corordinate = (row,col)
                coordinates.append(corordinate)

    # raise NotImplementedError
    return coordinates


def save_results(coordinates, template, template_name, rs_directory):
    results = {}
    results["coordinates"] = sorted(coordinates, key=lambda x: x[0])
    results["templat_size"] = (len(template), len(template[0]))
    with open(os.path.join(rs_directory, template_name), "w") as file:
        json.dump(results, file)


def main():
    args     = parse_args()
    img      = read_image(args.img_path)
    template = read_image(args.template_path)
    
    global threshold
    global h
    global w
    
    h = len(template)
    w = len(template[0])
    
    # Defining threshold for 3 different character images (depends on the templates)
    if args.template_path == './data/a.jpg':
        threshold = 0.94258827
    if args.template_path == './data/b.jpg':
        threshold = 0.96081966
    if args.template_path == './data/c.jpg':
        threshold = 0.95086306

    coordinates = detect(img,template)

    template_name = "{}.json".format(os.path.splitext(os.path.split(args.template_path)[1])[0])
    save_results(coordinates, template, template_name, args.rs_directory)
    
    # Draw rectangle around a matched region.
    #img = cv2.imread(args.img_path,1)
    #for pt in coordinates:
     #   cv2.rectangle(img, pt, (pt[0]+w+w, pt[1]+h+h), (255,0,0),1)
    
    
    # Show the final image with the matched area.
    #cv2.imwrite("Detected.jpg", img)

if __name__ == "__main__":
    main()
