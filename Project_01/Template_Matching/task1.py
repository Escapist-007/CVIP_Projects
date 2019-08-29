"""
   UB_ID : 50291708
   Name  : Md Moniruzzaman Monir

Edge Detection : (Due date: March 8th, 11: 59 P.M.)

The goal of this task is to experiment with two commonly used edge detection operator, i.e., Prewitt operator and Sobel operator,
and familiarize you with 'tricks', e.g., padding, commonly used by computer vision 'researchers'.

Please complete all the functions that are labelled with '# TODO'. Hints or steps are provided to make your lives easier.
Whem implementing the functions, comment the lines 'raise NotImplementedError' instead of deleting them.

As we have written lots of utility functions for you, you only need to write about 40 lines of code.

The functions defined in utils.py are building blocks you could use when implementing the functions labelled with 'TODO'.

I strongly suggest you to read the function "zero_pad" that is defined in utils.py. It is quite important!

Do NOT modify the code provided.
Do NOT use any API provided by opencv (cv2) and numpy (np) in your code.
Do NOT import any library (function, module, etc.).

How to run the  script : python task1.py --img_path="data/proj1-task1.jpg" --kernel="prewitt" or  python task1.py --img_path="data/proj1-task1.jpg" --kernel="sobel"

"""

import argparse
import copy
import os

import cv2
import numpy as np

import utils

# Prewitt operator
prewitt_x = [[1, 0, -1]] * 3 
prewitt_y = [[1] * 3, [0] * 3, [-1] * 3]

# Sobel operator
sobel_x = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
sobel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]


def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--img_path", type=str, default="",
        help="path to the image used for edge detection")
    parser.add_argument(
        "--kernel", type=str, default="sobel",
        choices=["prewitt", "sobel", "Prewitt", "Sobel"],
        help="type of edge detector used for edge detection")
    parser.add_argument(
        "--result_saving_directory", dest="rs_directory", type=str, default="./results/",
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


def write_image(img, img_saving_path):
    """Writes an image to a given path"""

    if isinstance(img, list):
        img = np.asarray(img, dtype=np.uint8)
    elif isinstance(img, np.ndarray):
        if not img.dtype == np.uint8:
            assert np.max(img) <= 1, "Maximum pixel value {:.3f} is greater than 1".format(np.max(img))
            img = (255 * img).astype(np.uint8)
    else:
        raise TypeError("img is neither a list nor a ndarray.")

    cv2.imwrite(img_saving_path, img)

    
    
def convolve2d(img, kernel):
    """Convolves a given image and a given kernel.
    
    Steps:
        (1) flips either the img or the kernel.

        (2) pads the img or the flipped img. This step handles pixels along the border of the img,
            and makes sure that the output img is of the same size as the input image.

        (3) applies the flipped kernel to the image or the kernel to the flipped image,using nested for loop.

    Args:
        img   : nested list (int), image.
        kernel: nested list (int), kernel.

    Returns:
        img_conv: nested list (int), image.
    """
    # TODO: DONE
    
    padding_layer = len(kernel)//2
    
    kernel = utils.flip2d(kernel)                              # flip the kernel
    img    = utils.zero_pad(img,padding_layer,padding_layer)   # padding the image
    
    row_count = len(img)  
    col_count = len(img[0])
   
    # output image having same size as the input image
    img_conv = []                                       
    
    zeros = [0]*(col_count-2*padding_layer)

    for i in range(row_count-2*padding_layer):
        img_conv.insert(0, [0 for value in enumerate(zeros)])
        
    kernel_h = len(kernel)
    kernel_w = len(kernel[0])

    for i in range(row_count - kernel_h + 1):
        for j in range(col_count - kernel_w + 1):
            # multiplying the cropped portion with kernel
            mult_result = utils.elementwise_mul( utils.crop(img,i,i+kernel_h,j,j+kernel_w), kernel) 
            sum = 0
            for p in range(len(mult_result)):
                for q in range(len(mult_result[p])):
                    sum += mult_result[p][q]
            img_conv[i][j] = sum
            
    #raise NotImplementedError
    return img_conv


def normalize(img):
    """Normalizes a given image.

    Hints:
        Normalize a given image using the following equation:
        
            normalized_img = frac{img - min(img)}{max(img) - min(img)},

        so that the maximum pixel value is 1 and the minimum pixel value is 0.

    Args:
        img: numpy.ndarray, np.uint8 --> image.

    Returns:
        normalized_img: numpy.ndarray, np.float32 --> normalized image.
    """
    # TODO: DONE

    img = np.asarray(img,dtype=np.float32)

    max_value = max([max(row) for row in img])
    min_value = min([min(row) for row in img])
    
    diff = (max_value - min_value)
    
    for r in range(len(img)):
        for c in range(len(img[r])):
            img[r][c] = (img[r][c]-min_value) / diff
    
    # raise NotImplementedError
    return img


def detect_edges(img, kernel, norm=True):
    """Detects edges using a given kernel.

    Args:
        img   : nested list (int), image.
        kernel: nested list (int), kernel used to detect edges.
        norm (bool): whether to normalize the image or not.

    Returns:
        img_edge: nested list (int), image containing detected edges.
    """
    # TODO: DONE
   
    # edge detection
    img_edge = convolve2d (img, kernel)
    
    # normalization
    if norm==True:
        img_edge = normalize(img_edge)
        
    #raise NotImplementedError
    return img_edge


def edge_magnitude(edge_x, edge_y):
    """Calculate magnitude of edges by combining edges along two orthogonal directions.

    Hints:
        Combine edges along two orthogonal directions using the following equation:

        edge_mag = sqrt(edge_x ** 2 + edge_y **).

        Make sure that you normalize the edge_mag, so that the maximum pixel value is 1. ***

    Args:
        edge_x: nested list (int), image containing detected edges along one direction.
        edge_y: nested list (int), image containing detected edges along another direction.

    Returns:
        edge_mag: nested list (int), image containing magnitude of detected edges.
    """
    
    # TODO: DONE
    edge_mag = copy.deepcopy(edge_x)
    
    for r in range(len(edge_mag)):
        for c in range(len(edge_mag[r])):
            x = edge_x[r][c] ** 2
            y = edge_y[r][c] ** 2
            edge_mag[r][c] = np.sqrt(x+y)

    edge_mag = normalize(edge_mag)
    #raise NotImplementedError
    return normalize(edge_mag)


def main():
    
    args = parse_args()

    img = read_image(args.img_path)

    if args.kernel in ["prewitt", "Prewitt"]:
        kernel_x = prewitt_x
        kernel_y = prewitt_y
    elif args.kernel in ["sobel", "Sobel"]:
        kernel_x = sobel_x
        kernel_y = sobel_y
    else:
        raise ValueError("Kernel type not recognized.")

    if not os.path.exists(args.rs_directory):
        os.makedirs(args.rs_directory)

    img_edge_x = detect_edges(img, kernel_x, False)
    img_edge_x = np.asarray(img_edge_x)

    write_image(normalize(img_edge_x), os.path.join(args.rs_directory, "{}_edge_x.jpg".format(args.kernel.lower())))

    img_edge_y = detect_edges(img, kernel_y, False)
    img_edge_y = np.asarray(img_edge_y)

    write_image(normalize(img_edge_y), os.path.join(args.rs_directory, "{}_edge_y.jpg".format(args.kernel.lower())))

    img_edges = edge_magnitude(img_edge_x, img_edge_y)
    write_image(img_edges, os.path.join(args.rs_directory, "{}_edge_mag.jpg".format(args.kernel.lower())))


if __name__ == "__main__":
    main()
