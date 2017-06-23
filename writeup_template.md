

---

# Udacity CarND Project 4 - Advanced Lane Finding Project #

---

The project covered the following content:

#### 1. Camera Calibration
* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.

#### 2. Image Processing
* Use gradients and color transforms to create a thresholded binary image.

#### 3. Image Perspective Transformation
* Apply a perspective transform to rectify binary image ("birds-eye view").

#### 4. Finding the Lane Lines
* Detect lane pixels and fit to find the lane boundary.

#### 5. Reading Lane Line Information
* Determine the curvature of the lane and vehicle position with respect to center.

#### 6. Visualiziation
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

#### 7. Video Pipeline
* Additional features of the video pipeline

#### 8. Outlook
* What could be further done to improve the results?

[//]: # (Image References)

[image1]: ./writeup_data/chessboard_original.jpg "Original Chessboard"
[image2]: ./writeup_data/chessboard_corner_pts.jpg "Corner Points Drawn on Chessboard"
[image3]: ./writeup_data/chessboard_undistorted.jpg "Undistorted Chessboard"
[image4]: ./writeup_data/pic_before_clahe.jpg "Original Image with Shadows"
[image5]: ./writeup_data/clahe.jpg "Equalized Image with Shadows"
[image6]: ./writeup_data/test_image.jpg "Test Image"
[image7]: ./writeup_data/hls.jpg "HLS Test Image"
[image8]: ./writeup_data/rgb.jpg "RGB Test Image"
[image9]: ./writeup_data/gradx.jpg "Sobel Grad X Test Image"
[image10]: ./writeup_data/grady.jpg "Sobel Grad Y Test Image"
[image11]: ./writeup_data/sobel_total.jpg "Combined Sobel Test Image"
[image12]: ./writeup_data/preprocessed.jpg "Combined Binary Image"
[image13]: ./writeup_data/warped.jpg "Warped Image"
[image14]: ./writeup_data/detected.jpg "Detected Lines Image"
[image15]: ./writeup_data/marker.jpg "Drawn Lines Image"
[image16]: ./writeup_data/result.jpg "Output"

[video1]: ./project_video.mp4 "Video"  

---

## 1. Camera Calibration

### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the file `camera_calibration.py`. 

I started by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the real world. Here I was assuming the 9x6 chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I created a list of the all 19 calibration images and went through this list one by one. The images were read and then converted to grayscale. The `cv2.findChessboardCorners()` function was used in order to detect the inner corners of the 9x6 chessboard. If the corners were successfully found the `objpoints` and `imgpoints` were appended to the different lists. The `cv2.drawChessboardCorners()` function was used to draw the corners onto a copy of the images and the image file was saved to verify the result. I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. The distortion correction was applied to one test image using the `cv2.undistort()` function in order to verify the result.

The following images show the distortion correction on one of the example pictures:

![alt text][image1]
![alt text][image2]
![alt text][image3]

The calibration data (camera matrix and distortion coefficients) was stored in a calibration file `calibration_pickle.p` (calibration_file/calibration_pickle.py) so that it could be easily used by the different pipelines.


All of the following steps were done in the pipeline for processing the single images as well as the video pipeline. Some additional features were added to the video pipeline later and will be discussed in section 7.

## 2. Image Processing

The images were opened and undistorted with the help of the saved correction parameters `mtx` (correction matrix) and `dist` (distortion coefficients). After that the picture was converted to YUV color space and changes in brightness (e.g. shadows) were corrected through a contrast limited adaptive histogram equalization (CLAHE) of the Y-channel. The built-in function `cv2.createCLAHE()` was used for this purpose. After that, the image was converted back to BGR color space.

Here is an example of an equalized image:

![alt text][image6]
![alt text][image5]

After the image was equalized copies of the image were modified with the help of two self-created funtions `hls_threshold` and `rgb_threshold`. `hls_threshold` converts the image to HLS color space and looks for values of the s channel that are in between two threshold values, which are defined as parameters of the function. A binary copy of the image is populated with this data, which means that all pixels of the image that are satisfying the threshold criteria are converted to 1, all other pixels have 0 value. The function `rgb_threshold` does the same thing with separate threshold parameters on the r channel and returns a binary image as well. Below you can see the result of `hls_threshold` and `rgb_threshold`, where the first-mentioned function seems to better pick up he yellow lines, while the white lines are better detected by the r channel in RGB color space.

Here is an example of an color thresholded image:

Test Image:
![alt text][image6]
Output of `hls_threshold`:
![alt text][image7]
Output of `rgb_threshold`:
![alt text][image8]

Another two copies of the image were modified by the function `abs_sobel_thresh` that calculates a thresholded binary picture of the absolute sobel function in x and y direction. The image was converted to grayscale and a bilateral noise filter was applied. This filter was chosen instead of a gaussian blur because it removes noise from the image without significantly effecting the sharpness of the edges. This helped to minimize the detected edges and lines caused by shadows and patterns on the asphalt while keeping the lane lines visible. Two different thresholds were chosen for the different directions with a lower threshold in x direction as we are more interested in lines closer to vertical than horizontal. Both output images were combined so that a picture was returned that only showed the pixels that were detected in x and y direction.

Here is an example of the output image after sobel calculation:

Test Image:
![alt text][image6]
Sobel in x direction:
![alt text][image9]
Sobel in y direction:
![alt text][image10]
Combined sobel:
![alt text][image11]

All three binary images (hls, rgb, sobel) were combined to show all detected and thresholded pixels in one image called `preprocessImage`.

Combined image `preprocessImage`:
![alt text][image12]


## 3. Image Perspective Transform

For the perspective transform was defined by source (`src`) and destination (`dst`) image points. The `src` image points form a trapezoid containing the lane lines during a straight section and the `dst` points are a centered rectangle over the full height of the image (720 pixels).  The transformation matrix `M` and the inverse version `Minv` was calculated with help of the built-in function `cv2.cv2.getPerspectiveTransform()` and the above mentioned source and destiantion points. With the function `cv2.warpPerspective()` and the matrix `M` the preprocessed Image was transformed into a birds eye view perspective. This makes it easier to analyze the curvature of the lane lines.

Processed image transformed into birds eye perspective:
![alt text][image13]


## 4. Finding the Lane Lines

In order to find the lane lines I used the sliding window approach. I took a histogram of the bottom half of the image in order to find the starting position. The peak on the left and right side (highest pixel density) defined the starting position of the first window. I decided to use 14 windows with a width of 80 pixels for the first window and 60 pixels for all windows after. The minimum number of pixels was set to 50. I experimented with different values, but achieved the best results overall with those settings. The windows were drawn over the picture and the found pixels were highlighted in order to verify the output.

Sliding window approach to detect lane lines:
![alt text][image14]


## 5. Reading Lane Line Information

The result was the arrays of indices for both lines as well as the position of the lane line pixels in x and y direction. Those values were used to fit a second order polynomial to each line with the function `np.polyfit()` which was saved in the variables `left_fit` and `right_fit`. Based on that result the x values of the polinomial was calculated and the data was stored in the variables `left_fitx` and `right_fitx`. Those values were used to calculate the radii of the left line, the right line and (a newly created) middle line as well as the direction of the curvature (straight, left or right) and the points for the new middle line. The self-built function `calculate_middle_line()` shows the calculations.


## 6. Visualization

In order to visualize the results of the calculated polinomial and the overall performance of the lane detection algorithm I created two polynoms around the two second order polinomials and filled those with color (red and blue). I also colored the space in between those two areas with a green color and displayed the middle line with white pixels. The built-in functions `cv2.fillPoly()` and `cv2.polylines()` were used for that purpose. Those markers were than drawn into a new picture (still in birds eye view) and looked like the following:

![alt text][image15]

The picture was then warped back to the original perspective with the inverse transformation matrix `Minv` and the function `cv2.warpPerspective()`. This picture was then overlayed with the original image by using `cv2.addWeighted()` in order to highlight the different detected zones. Additional infomation was included by adding text to the image using `cv2.putText()`. The result looks as follows:

![alt text][image16]

All pictures from the different stages were saved in the output image folder (/output_images) in order to evaluate the performance of the different steps such as sobel, hls and rgb color thresholding, line detection etc. 


## 7. Video Pipeline

For the video pipeline I used a new file called `video_gen.py`. The video was opened and the single images processed by using the functions `VideoFileClip()` and `fl_image()` from the `moviepy.editor` library. Additionally a class for storing the values from the last frame (`Last_Frame()`) was created that also contained a class `Line()` for the values for the left and right lane line parameters. ANother class called `Error()` was created in order to store error information. In addition to the sliding window approach I used the approach to search around the last frame's lane line position. The margin for this search window in x direction was set to 80 pixels after a couple trials. This gave good results to find enough lane line pixels without including to much noise around the lane lines. You can find the code strating in line 335 in `video_gen.py`.

In order to avoid any misinterpretation and wrong curvature calculations I added some sanity checks that were applied to each image. The following was verified for each image:
* It was checked if the lane lines are in a certain distance or if they are to close together or too far apart.
* The lines were checked for parallelism and if the lines were further than a certain margin over or under the average distance an error flag was set.
* If the lane changes direction abruptly from one frame to the other without going through the "straight" section (radius > 3,000m) an error flag was set.
* It was verified that the curvature was over a certain minimum radius (150m for the highway test video).
* Both lane lines were checked individually if they changed there curvature value abruptly from one frame to the other. If the change was higher than `change_factor` (set to 3) a specific error flag for this specific line was set. If the other line was considered to be "good" based on all other sanity checks the "bad" line was replaced with a copy of the "good" line in average distance (left or right) to the "good" line. No error flag was set in this situation.

If at least one sanity check failed the error counter `tracking_errors.counter` was increased by one, the error mode was added to a error list (`tracking_errors.add_errors_to_list()`), the values of the old frame were used, and all errors flags were reset to "False". If no error occured the error counter was reset to 0, the error list was emptied and all current values were saved in the variables for the parameters of the last frame. After that the output was smoothened to reduce the effect of "jumping lines" by using a FIFO buffer with a `collections.deque()` container. The buffer size was defined in the variable `last_frame.buffer_size` and was set to 8. This provided a good smoothening effect while still being able to adapt quickly to curvature changes. The image was visualized the same way as before, but now based on the smoothened results.

All different steps were again saved in images and this time added as smaller images/videos that were overlayed with the final result. This was very important for tuning, trouble shooting and understanding of the influences of the differnent parameters. The result is the video `output_video_highway.mp4`.


## 8. Outlook

In order to make the algorithm more versatile and robust additional information could have been used to automatically adjust certain parameters, such as the different thresholds, field of view, error flag limitations etc. Those parameters could have been speed of the vehicle, type of street the vehicle is currently on (e.g. highway vs. curvy road), time of the day, weather and many more. This would allow the algorithm to automatically adjust to different conditions.
