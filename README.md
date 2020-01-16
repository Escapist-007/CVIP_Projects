# Spring 2019
<p align="center">
<img src="images/ub.png" alt="ub_logo.jpg" width="100" height="100"> <br>
  <b> CSE-573: Computer Vision & Image Processing </b>
</p>

### [Edge Detection](Project_01/Edge_Detection) :
<img src="images/bar.jpg" alt="bar.jpg" width="1100" height="3"> <br>

**Problem:** 
Detect edges of a grayscale image. Do not use any `API` provided by `opencv` (cv2) and `numpy`(np) in your code (except “np.sqrt()”).

**Approach:**
- The project applied `Sobel` and `Prewitt` filters to detect edges in a given image
- Implemented common image processing tasks : 
  - padding
  - convolution
  - correlation
  - normalization etc.
  
**Sample input and output:** 

Input image: <br>

<p align="center">
<img src="Project_01/Edge_Detection/data/proj1-task1.jpg" alt="input_image.jpg">
</p>

Output image: <br>

Edge detection using `Prewitt` filter: 

<img src="Project_01/Edge_Detection/results/prewitt_edge_x.jpg" alt="prewitt_x.jpg"><img src="Project_01/Edge_Detection/results/prewitt_edge_y.jpg" alt="prewitt_y.jpg"><img src="Project_01/Edge_Detection/results/prewitt_edge_mag.jpg" alt="prewitt_mag.jpg">

Edge detection using `Sobel` filter: 

<img src="Project_01/Edge_Detection/results/sobel_edge_x.jpg" alt="sobel_x.jpg"><img src="Project_01/Edge_Detection/results/sobel_edge_y.jpg" alt="sobel_y.jpg"><img src="Project_01/Edge_Detection/results/sobel_edge_mag.jpg" alt="sobel_mag.jpg">


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Back To The Top](#spring-2019)


### [Template Matching](Project_01/Template_Matching) :
<img src="images/bar.jpg" alt="gray.jpg" width="1100" height="3"> <br>

**Problem:** 
`Character Detection`: Find a specific character in a given image using template matching algorithms.

**Approach:**
- The project applied **Template matching algorithm** to detect a specific character (ex. a/b/c) in a given image
- Implemented `NCC (Normalized Cross Correlation)` for matching the template with the given image

**Sample input and output:** 

Input image: <br>

<p align="center">
<img src="Project_01/Template_Matching/data/proj1-task2.jpg" alt="input_image.jpg">
</p>


**Templates:**
<img src="Project_01/Template_Matching/data/a.jpg" alt="a.jpg">
<img src="Project_01/Template_Matching/data/b.jpg" alt="b.jpg">
<img src="Project_01/Template_Matching/data/c.jpg" alt="c.jpg">

Output image: <br>
**detecting a**

<img src="Project_01/Template_Matching/output_demo/Detected_a.jpg">

**detecting b**

<img src="Project_01/Template_Matching/output_demo/Detected_b.jpg">



&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Back To The Top](#spring-2019)


### [Panorama/Image Stitching](Project_02) :
<img src="images/bar.jpg" alt="gray.jpg" width="1100" height="3"> <br>

**Problem:** 
`Image Stitching`: Create a panoramic image from at most 3 images. Overlap of the given images will be at least 20% and not more than 50%. Any API provided by OpenCV could be used, except “`cv2.findHomography()`” and APIs that have “stitch”, “Stitch”, “match” or “Match” in their names, e.g., “cv2.BFMatcher()” and “cv2.Stitcher.create()”.

**Approach:**
- Keypoints detection and 128 bit feature vector computation using `SIFT`
- Homography matrix generation using `SVD`
- Implemented `RANSAC` algorithm for finding the best Homography matrix
- Stitched all images


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Project Report](Project_02/Report.pdf)


**Sample input and output:** 

Input image: <br>

<img src="Project_02/data/nevada/nevada3.jpg" alt="nevada3.jpg" width="300" height="250"><img src="Project_02/data/nevada/nevada4.jpg" alt="nevada4.jpg" width="300" height="250"><img src="Project_02/data/nevada/nevada5.jpg" alt="nevada5.jpg" width="250" height="250">

Output image: <br>
<img src="Project_02/data/nevada/panorama.jpg" alt="nevada_panoroma.jpg">


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Back To The Top](#spring-2019)


### [Face Detection in the Wild](Project_03) :
<img src="images/bar.jpg" alt="gray.jpg" width="1100" height="3"> <br>

**Problem:** 
Implement the `Viola-Jones` face detection algorithm which is capable of detecting frontal faces in real time and is regarded as a milestone in the development of computer vision. Any APIs provided by OpenCV that have “cascade”, “Cascade”, “haar” or “Haar” functionality can not be used. Using any APIs that implement part of Viola-Jones algorithm directly, e.g., an API that computes integral image, will result in a deduction of 10% − 100% of the maximum possible points of this project

**Approach:**
- Used `FDDB` dataset to train the model with 'face images and `CBCL` dataset to train with 'non-face images'
- Implemented `integral image` calculation
- `Adaboost` implementation
- Developed `CASCADING` to reject non-face region quickly


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Project Report](Project_03/Report.pdf)


**output:** <br>

<img src="Project_03/sample_output/1.png" width="300" height="250"><img src="Project_03/sample_output/2.png" width="300" height="250"><img src="Project_03/sample_output/3.png" width="250" height="250">


---
## License

[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

- **[MIT license](http://opensource.org/licenses/mit-license.php)**


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Back To The Top](#spring-2019)
