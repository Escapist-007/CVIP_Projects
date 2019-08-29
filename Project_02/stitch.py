"""
    UB_ID : 50291708
    Name  : Md Moniruzzaman Monir
    
    Image Stitching : (Due date: April 8th, 11:59 P.M.)
    
"""
import cv2
import os
import sys
import glob
import numpy as np
import random as rand
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist

UBIT = '50291708'
np.random.seed(sum([ord(c) for c in UBIT]))


def detect_and_draw_keypoints(image,i):
    """
        Draw keypoints using SIFT method.

        Args:
            image: Input image to detect and draw the keypoints.
            i    : Int/Float value to indicate the output image

        Returns:
             None
    """
    sift = cv2.xfeatures2d.SIFT_create()
    kps  = sift.detect(image, None)   # only detect key points, kps is a list keypoins
    cv2.imwrite('out'+ str(i) +'_kp.jpg', cv2.drawKeypoints(image,kps,image,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))


def detect_and_compute_keypoints(image):
    """
        Detect keypoints and Descriptor of every keypoint using SIFT method.
        
        Args:
            image: Input image to detect and compute the keypoints.
        
        Returns:
            kps: List of keypoints
            desc : descriptors of the keypoints
    """
    sift = cv2.xfeatures2d.SIFT_create()
    kps, desc = sift.detectAndCompute(image,None) # "kps" is a list of keypoints and "desc" is a numpy array of shape "Number_of_Keypoints × 128".
    
    return kps, desc

                                                                    ##    KEYPOINT MATCHING  ##

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html#scipy.spatial.distance.cdist

def matching_keypoints_sqeuclidian(kps1, kps2, desc1, desc2):
    """
        Find matching descriptors and corresponding keypoins between 2 images. 
        Descriptor matching is done using "scipy.spatial.distance.cdist()" method.
        Here, Squared Euclidian Distance is used as the distance metric.
    """
    pairwiseDistances = cdist(desc1, desc2, 'sqeuclidean')
    threshold   = 7000
    
    points_in_img1 = np.where(pairwiseDistances < threshold)[0]
    points_in_img2 = np.where(pairwiseDistances < threshold)[1]
    
    coordinates_in_img1 = np.array([kps1[point].pt for point in points_in_img1])
    coordinates_in_img2 = np.array([kps2[point].pt for point in points_in_img2])
    
    return np.concatenate( (coordinates_in_img1, coordinates_in_img2) , axis=1 )


def matching_keypoints_hamming(kps1, kps2, desc1, desc2):
    """
        Find matching descriptors and corresponding keypoins between 2 images.
        Descriptor matching is done using "scipy.spatial.distance.cdist()" method.
        Squared Euclidian Distance is used as the distance metric.
        """
    pairwiseDistances = cdist(desc1, desc2, 'hamming')  # normalized hamming distance
    
    # Atleast 20% overlap
    threshold   = 0.20
    
    # Return a list of 2 elements : 1st element contains all row number and 2nd element contain all col number
    points_Row_Col = np.where(pairwiseDistances < threshold)
    
    if len(points_Row_Col[0]) < 20:
        threshold = 0.25
        points_Row_Col = np.where(pairwiseDistances < threshold)
    
    if len(points_Row_Col[0]) < 20:
        threshold = 0.28
        points_Row_Col = np.where(pairwiseDistances < threshold)
    
    if len(points_Row_Col[0]) < 20:
        threshold = 0.31
        points_Row_Col = np.where(pairwiseDistances < threshold)

    if len(points_Row_Col[0]) < 20:
        threshold = 0.34
        points_Row_Col = np.where(pairwiseDistances < threshold)

    if len(points_Row_Col[0]) < 20:
        threshold = 0.37
        points_Row_Col = np.where(pairwiseDistances < threshold)

    if len(points_Row_Col[0]) < 20:
        threshold = 0.40
        points_Row_Col = np.where(pairwiseDistances < threshold)

    if len(points_Row_Col[0]) < 20:
        threshold = 0.43
        points_Row_Col = np.where(pairwiseDistances < threshold)

    if len(points_Row_Col[0]) < 20:
        threshold = 0.46
        points_Row_Col = np.where(pairwiseDistances < threshold)

    if len(points_Row_Col[0]) < 20:
        threshold = 0.49
        points_Row_Col = np.where(pairwiseDistances < threshold)
    
    if len(points_Row_Col[0]) < 20:
        threshold = 0.52
        points_Row_Col = np.where(pairwiseDistances < threshold)

    if len(points_Row_Col[0]) < 20:
        threshold = 0.55
        points_Row_Col = np.where(pairwiseDistances < threshold)

    if len(points_Row_Col[0]) < 20:
        threshold = 0.58
        points_Row_Col = np.where(pairwiseDistances < threshold)
    
    if len(points_Row_Col[0]) < 20:
        threshold = 0.61
        points_Row_Col = np.where(pairwiseDistances < threshold)
    
    if len(points_Row_Col[0]) < 20:
        threshold = 0.64
        points_Row_Col = np.where(pairwiseDistances < threshold)

    if len(points_Row_Col[0]) < 20:
        threshold = 0.8
        points_Row_Col = np.where(pairwiseDistances < threshold)
    
    print(threshold)
    
    points_in_img1   = points_Row_Col[0]  # row numbers represent points in the 1st image
    points_in_img2   = points_Row_Col[1]  # col numbers represent points in the 2nd image

    # List of tuples as each coordinate (x,y) is a tuple
    coordinates_in_img1 = []
    coordinates_in_img2 = []

    for point in points_in_img1:
        coordinates_in_img1.append(kps1[point].pt)

    for point in points_in_img2:
        coordinates_in_img2.append(kps2[point].pt)
    
    return np.concatenate( ( np.array(coordinates_in_img1), np.array(coordinates_in_img2) ), axis=1)




                                                                ##  Homography Matrix and RANSAC Algorithm  ##


def ransac_algo(matchingPoints,totalIteration):
    
    # Ransac parameters
    highest_inlier_count = 0
    best_H = []
    
    # Loop parameters
    counter = 0
    while counter < totalIteration:
        counter = counter + 1
        # Select 4 points randomly
        secure_random  = rand.SystemRandom()
        
        matachingPair1 = secure_random.choice(matchingPoints)
        matachingPair2 = secure_random.choice(matchingPoints)
        matachingPair3 = secure_random.choice(matchingPoints)
        matachingPair4 = secure_random.choice(matchingPoints)
        
        fourMatchingPairs=np.concatenate(([matachingPair1],[matachingPair2],[matachingPair3],[matachingPair4]),axis=0)
        
        # Finding homography matrix for this 4 matching pairs
        # H = get_homography(fourMatchingPairs)

        points_in_image_1 = np.float32(fourMatchingPairs[:,0:2])
        points_in_image_2 = np.float32(fourMatchingPairs[:,2:4])
        
        H = cv2.getPerspectiveTransform(points_in_image_1, points_in_image_2)
        
        rank_H = np.linalg.matrix_rank(H)
        
        # Avoid degenrate H
        if rank_H < 3:
            continue
        
        # Calculate error for each point using the current homographic matrix H
        total_points = len(matchingPoints)
        
        points_img1 = np.concatenate( (matchingPoints[:, 0:2], np.ones((total_points, 1))), axis=1)
        points_img2 = matchingPoints[:, 2:4]
        
        correspondingPoints = np.zeros((total_points, 2))
        
        for i in range(total_points):
            t = np.matmul(H, points_img1[i])
            correspondingPoints[i] = (t/t[2])[0:2]

        error_for_every_point = np.linalg.norm(points_img2 - correspondingPoints, axis=1) ** 2

        inlier_indices = np.where(error_for_every_point < 0.5)[0]
        inliers        = matchingPoints[inlier_indices]
    
        curr_inlier_count = len(inliers)
      
        if curr_inlier_count > highest_inlier_count:
            highest_inlier_count = curr_inlier_count
            best_H = H.copy()

    return best_H


def main():
    
    directory = sys.argv[1]
    imageDir  = directory + '/*.jpg'
    
    images         = [cv2.imread(file,0) for file in glob.glob(imageDir)]
    colorImages    = [cv2.imread(file,1) for file in glob.glob(imageDir)]
    
    if len(images)>3:
        print("Can read atmost 3 images")
        return
    if len(images)==0:
        print("No images with .jpg extension")
        return
    if len(images)==1:
        print("Single image. Can't create a panorama")
        return
    if len(images)==2:
        # SIFT feature detection
        sift = cv2.xfeatures2d.SIFT_create()
        
        kps1, desc1 = sift.detectAndCompute(images[0],None)
        kps2, desc2 = sift.detectAndCompute(images[1],None)

        # Stitching image 1 and image 2
        H12 = ransac_algo(matching_keypoints_sqeuclidian(kps1,kps2, desc1,desc2), 1000)
        
        result = cv2.warpPerspective(colorImages[0], H12 ,
                                     ( int(colorImages[0].shape[1] + colorImages[1].shape[1]*0.8),
                                       int(colorImages[0].shape[0] + colorImages[1].shape[0]*0.4) )
                                     )
            
        result[0:colorImages[1].shape[0], 0:colorImages[1].shape[1]] = colorImages[1]
                                     
        cv2.imwrite( directory + '/panorama.jpg', result)
    
        # Resizing the final panorama
        black = np.zeros(3)
        colorPan = cv2.imread(directory + '/panorama.jpg', 1)
        x_max = 0
        y_max = 0
        
        for i in range(colorPan.shape[0]):
            for j in range(colorPan.shape[1]):
                pixel_value = colorPan[i, j, :]
                if not np.array_equal(pixel_value, black):
                    if j > x_max:
                        x_max = j
                    if i > y_max:
                        y_max = i

        crop_img = colorPan[0:y_max,0:x_max, :]
        os.remove(directory + '/panorama.jpg')
        cv2.imwrite( directory + '/panorama.jpg', crop_img)
              
    else:
        # SIFT feature detection
        sift = cv2.xfeatures2d.SIFT_create()
        
        kps1, desc1 = sift.detectAndCompute(images[0],None)
        kps2, desc2 = sift.detectAndCompute(images[1],None)
        kps3, desc3 = sift.detectAndCompute(images[2],None)

        a12 = matching_keypoints_sqeuclidian(kps1,kps2,desc1,desc2)
        a13 = matching_keypoints_sqeuclidian(kps1,kps3,desc1,desc3)
        a23 = matching_keypoints_sqeuclidian(kps2,kps3,desc2,desc3)
        
        totalMatch_img1 = len(a12) + len(a13)
        totalMatch_img2 = len(a12) + len(a23)
        totalMatch_img3 = len(a13) + len(a23)
        
        if totalMatch_img1 >= totalMatch_img2 and totalMatch_img1  >= totalMatch_img3:
            centerIdx = 0
        elif totalMatch_img2 >= totalMatch_img1 and totalMatch_img2 >= totalMatch_img3:
            centerIdx = 1
        else:
            centerIdx = 2

        if centerIdx==0:
            # swap 1st and 2nd images
            temp = images[0]
            images[0] = images[1]
            images[1] = temp

            tempC = colorImages[0]
            colorImages[0] = colorImages[1]
            colorImages[1] = tempC
        
        elif centerIdx==2:
            # swap 2nd and 3rd images
            temp = images[2]
            images[2] = images[1]
            images[1] = temp
            
            tempC = colorImages[2]
            colorImages[2] = colorImages[1]
            colorImages[1] = tempC
        else:
            pass

        # For new ordering
        kps1, desc1 = sift.detectAndCompute(images[0],None)
        kps2, desc2 = sift.detectAndCompute(images[1],None)
        kps3, desc3 = sift.detectAndCompute(images[2],None)


                                                                    ## FORWARD STITCHING

        # Stitching image 1 and image 2
        H12 = ransac_algo(matching_keypoints_sqeuclidian(kps1,kps2, desc1,desc2), 1000)

        result = cv2.warpPerspective(colorImages[0], H12 ,
                                     ( int(colorImages[0].shape[1] + colorImages[1].shape[1]*0.8),
                                       int(colorImages[0].shape[0] + colorImages[1].shape[0]*0.4) )
                                     )

        result[0:colorImages[1].shape[0], 0:colorImages[1].shape[1]] = colorImages[1]

        cv2.imwrite( directory + '/panorama12.jpg', result)


        # Stitching image 12 and image 3
        image12 = cv2.imread(directory + '/panorama12.jpg',0)
        color12 = cv2.imread(directory + '/panorama12.jpg',1)

        kps12, desc12 = sift.detectAndCompute(image12,None)

        H23 = ransac_algo(matching_keypoints_sqeuclidian(kps12,kps3, desc12,desc3), 1000)

        result = cv2.warpPerspective(color12, H23 ,
                                     ( int(color12.shape[1] + colorImages[2].shape[1]*0.8),
                                       int(color12.shape[0] + colorImages[2].shape[0]*0.4) )
                                     )

        result[0:colorImages[2].shape[0], 0:colorImages[2].shape[1]] = colorImages[2]

        cv2.imwrite( directory + '/panorama123.jpg', result)
        os.remove(directory + '/panorama12.jpg')


                                                                ## BACKWARD STITCHING

        # Stitching image 3 and image 2
        H32 = ransac_algo(matching_keypoints_sqeuclidian(kps3,kps2, desc3,desc2), 1000)

        result = cv2.warpPerspective(colorImages[2], H32 ,
                                     ( int(colorImages[2].shape[1] + colorImages[1].shape[1]*0.8),
                                       int(colorImages[2].shape[0] + colorImages[1].shape[0]*0.4) )
                                     )

        result[0:colorImages[1].shape[0], 0:colorImages[1].shape[1]] = colorImages[1]

        cv2.imwrite( directory + '/panorama32.jpg', result)

        # Stitching image 32 and image 1
        image32 = cv2.imread(directory + '/panorama32.jpg',0)
        color32 = cv2.imread(directory + '/panorama32.jpg',1)

        kps32, desc32 = sift.detectAndCompute(image32,None)

        H21 = ransac_algo(matching_keypoints_sqeuclidian(kps32,kps1, desc32, desc1), 1000)

        result = cv2.warpPerspective(color32, H21 ,
                                     ( int(color32.shape[1] + colorImages[0].shape[1]*0.8),
                                       int(color32.shape[0] + colorImages[0].shape[0]*0.4) )
                                     )
            
        result[0:colorImages[0].shape[0], 0:colorImages[0].shape[1]] = colorImages[0]

        cv2.imwrite( directory + '/panorama321.jpg', result)
        os.remove(directory + '/panorama32.jpg')


                                            ##  DECIDE THE BEST PANORAMA BASED ON THE NUMBER OF TOTAL BLACK PIXELS

        black = np.zeros(3)
        color123 = cv2.imread(directory + '/panorama123.jpg', 1)
        color321 = cv2.imread(directory + '/panorama321.jpg', 1)
        
        count123 = 0
        count321 = 0
        
        for i in range(color123.shape[0]):
            for j in range(color123.shape[1]):
                pixel_value = color123[i, j, :]
                if np.array_equal(pixel_value, black):
                    count123 = count123 + 1

        for i in range(color321.shape[0]):
            for j in range(color321.shape[1]):
                pixel_value = color321[i, j, :]
                if np.array_equal(pixel_value, black):
                    count321 = count321 + 1

        if count123 < count321:
           os.remove(directory + '/panorama321.jpg')
           os.rename(directory + '/panorama123.jpg', directory +'/panorama.jpg')
        else:
           os.remove(directory + '/panorama123.jpg')
           os.rename(directory + '/panorama321.jpg',directory + '/panorama.jpg')

        # Resizing the final panorama
        colorPan = cv2.imread(directory + '/panorama.jpg', 1)
        x_max = 0
        y_max = 0

        for i in range(colorPan.shape[0]):
            for j in range(colorPan.shape[1]):
                pixel_value = colorPan[i, j, :]
                if not np.array_equal(pixel_value, black):
                    if j > x_max:
                       x_max = j
                    if i > y_max:
                       y_max = i
    
        crop_img = colorPan[0:y_max,0:x_max, :]
        os.remove(directory + '/panorama.jpg')
        cv2.imwrite( directory + '/panorama.jpg', crop_img)


if __name__ == "__main__":
    main()










