import cv2
import numpy as np
import glob
import pickle
from tracker import tracker
import collections

from moviepy.editor import VideoFileClip
from IPython.display import HTML

import matplotlib.pyplot as plt

# Define conversions in x and y from pixels space to meters
ym_per_pix = 22/720 # meters per pixel in y dimension
xm_per_pix = 4/570  # meters per pixel in x dimension

# read in the saved objpoints and imgpoints
dist_pickle = pickle.load(open("./calibration_file/calibration_pickle.p", "rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]


# Line class to store certain features and perform sanity checks
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # polynomial fit of last "good" frame
        self.recent_fit = None
        # polynomial fitx of last "good" frame
        self.recent_fitx = None
        # radius of curvature of the line in meters
        self.last_radius_m = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None 


class Last_Frame():
    def __init__(self):
        self.recent_radius = None
        self.recent_direction = "straight"
        self.recent_middle_pts = None
        self.buffer_size = 15 # 1 = no buffer
        self.average_left_fitx = collections.deque([], self.buffer_size)
        self.average_right_fitx = collections.deque([], self.buffer_size)
        self.average_lane_distance = collections.deque([], self.buffer_size)       
        self.leftx_base = None
        self.rightx_base = None
                
        self.left_line = Line()
        self.right_line = Line()


class Error():
    def __init__(self):
        # Reset tracking mode
        self.lost_tracking = False # Set to True for the first frame
        # Flag for first frame exception
        self.first_frame = True # Set to True for the first frame
        # Counter for counting frames with at least one error
        self.counter = 0 # Initialize with 0
        # Maximum number of errors allowed before reset
        self.max_errors = 5
        # Error for not deteting lane lines in the last images
        self.no_lines = False # Set to True for the first frame
        # Error for curvature being out of range
        self.curvature = False
        # Error for lines not being parallel
        self.parallel = False
        # Error for distance between the alne lines
        self.distance = False
        # Error for abrupt change of direction
        self.direction = False
        # Error list for tracking the occuring error modes
        self.error_list = []

    def add_errors_to_list(self):
        error_message = " " 
        if self.curvature == True:
            error_message += "curvature "
        if self.parallel == True:
            error_message += "parallel "
        if self.distance == True:
            error_message += "distance "
        if self.direction == True:
            error_message += "direction "
        error_message += "\n"
        self.error_list.append(error_message)

  
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0,255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply noise filter
    gray = cv2.bilateralFilter(gray, 5, 100, 100)
    
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))    
    binary_output = np.zeros_like(scaled_sobel)
    # apply threshold
    binary_output[(scaled_sobel >= thresh[0])&(scaled_sobel <= thresh[1])] = 1
    return binary_output


"""
def mag_threshold(image, sobel_kernel=3, mag_thresh=(0,255)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    gradmag = np.sqrt(sobelx**2 - sobely**2)
    scale_factor = np.max(gradmag)*255    
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    binary_output = np.zeros_like(gradmag)
    # apply threshold
    binary_output[(gradmag >= mag_thresh[0])&(gradmag <= mag_thresh[1])] = 1
    return binary_output


def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    # calculate gradient direction
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    with np.errstate(divide='ignore', invalid='ignore'):
        absgraddir = np.absolute(np.arctan(sobely/sobelx))
        binary_output = np.zeros_like(absgraddir)
        # apply threshold
        binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
        
    return binary_output
"""


def hls_threshold(image, sthresh=(0,255)):
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    s_channel = hls[:,:,2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= sthresh[0]) & (s_channel <= sthresh[1])] = 1

    output = np.zeros_like(s_channel)
    output[(s_binary == 1)] = 1
    
    return output


def rgb_threshold(image, rthresh=(0,255)):
    r_channel = image[:,:,2] # BGR image format
    r_binary = np.zeros_like(r_channel)
    r_binary[(r_channel >= rthresh[0]) & (r_channel <= rthresh[1])] = 1

    output = np.zeros_like(r_channel)
    output[(r_binary == 1)] = 1
    
    return output


def sliding_window(binary_warped, out_img):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
    # Find the peak of the left and right halves of the histogram, which will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)

    # Store base values as starting points for next frames
    last_frame.leftx_base = np.argmax(histogram[:midpoint])
    last_frame.rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 14
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = last_frame.leftx_base
    rightx_current = last_frame.rightx_base
    # Set the width of the windows +/- margin for first window
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        # Change the window margin for all other windows (besides the first)
        margin = 80

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Generate colored lane line image
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # save polinomial fit from current frame
    last_frame.left_line.recent_fit = left_fit
    last_frame.right_line.recent_fit = right_fit

    # Set back error after first frame was processed
    #tracking_errors.lost_tracking = False

    return left_fitx, right_fitx, ploty, out_img


def search_with_margin(binary_warped, out_img):
    # Assume you now have a new warped binary image 
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # load polinomial fit from last frame
    left_fit = last_frame.left_line.recent_fit
    right_fit = last_frame.right_line.recent_fit

    # margin for searching around expected location
    margin = 80
    # identify new non zero pixels around the expected location 
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a new second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # save new polinomial fit from current frame
    last_frame.left_line.recent_fit = left_fit
    last_frame.right_line.recent_fit = right_fit
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Generate colored lane line image
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    return left_fitx, right_fitx, ploty, out_img


def calculate_middle_line(left_fitx, right_fitx, ploty):
    # Calculate line in the middle
    bottom_center = left_fitx[-1] + (right_fitx[-1] - left_fitx[-1])/2
    lower_third = left_fitx[int(len(left_fitx)/3)] + (right_fitx[int(len(right_fitx)/3)] - left_fitx[int(len(left_fitx)/3)])/2
    upper_third = left_fitx[int(2*len(left_fitx)/3)] + (right_fitx[int(2*len(right_fitx)/3)] - left_fitx[int(2*len(left_fitx)/3)])/2
    top_center = left_fitx[0] + (right_fitx[0] - left_fitx[0])/2
    center_pts_x = [bottom_center, lower_third, upper_third, top_center]
    center_pts_y = [ploty[-1], ploty[int(len(ploty)/3)], ploty[int(2*len(ploty)/3)], ploty[0]]
    middle_fit = np.polyfit(center_pts_y, center_pts_x, 2)
    middle_fitx = middle_fit[0]*ploty**2 + middle_fit[1]*ploty + middle_fit[2]
    middle_pts = np.array([np.transpose(np.vstack([middle_fitx, ploty]))])

    # fit new polynomials to x,y in real world space
    middle_fit_real_world = np.polyfit(ploty*ym_per_pix, middle_fitx*xm_per_pix, 2)
    left_fit_real_world = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_real_world = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    # calculate radius for a position 260 pixels up from the bottom frame
    y_eval = np.max(ploty)-260 
    # calculate radius for real world space (meters)
    middle_curverad_m = ((1 + (2*middle_fit_real_world[0]*y_eval*ym_per_pix + middle_fit_real_world[1])**2)**1.5) / np.absolute(2*middle_fit_real_world[0])
    left_curverad_m = ((1 + (2*left_fit_real_world[0]*y_eval*ym_per_pix + left_fit_real_world[1])**2)**1.5) / np.absolute(2*left_fit_real_world[0])
    right_curverad_m = ((1 + (2*right_fit_real_world[0]*y_eval*ym_per_pix + right_fit_real_world[1])**2)**1.5) / np.absolute(2*right_fit_real_world[0])

    # calculate direction
    c_distance = middle_fitx[0] - middle_fitx[-1]
    straight_curvature = 3000
    if (middle_curverad_m > straight_curvature):
        direction = "straight"
    else:
        if (c_distance > 0):
            direction = "right"
        else:
            direction = "left"
    
    return left_curverad_m, middle_curverad_m, right_curverad_m, direction, middle_pts


def check_lane_lines(left_fitx, right_fitx, direction, average_lane_distance):
   
    # detected lanes are too wide or too narrow
    lane_distance = np.mean(right_fitx - left_fitx)
    if ((lane_distance > (4.5 / xm_per_pix)) or (lane_distance < (3.5 / xm_per_pix))):
        tracking_errors.distance = True

    # detected curves are not parallel (compare with average lane line width)
    if (tracking_errors.first_frame == False):
        lane_dist = average_lane_distance
        if (((np.max(right_fitx - lane_dist - left_fitx) > (0.5 / xm_per_pix)) or np.min(right_fitx - lane_dist - left_fitx)< (-0.5 / xm_per_pix))):
            tracking_errors.parallel = True
            print ("Lines are not parallel.")
            print (xm_per_pix * np.max(right_fitx - lane_dist - left_fitx))
        
    # detected curve changes direction abruptly
    if (last_frame.recent_direction == "right" and direction == "left" or last_frame.recent_direction == "left" and direction == "right"):
        tracking_errors.direction  = True

    return


def check_line_curvature(left_curverad_m, middle_curverad_m, right_curverad_m, left_fitx, right_fitx):
     
    # check minimum curvature of lane
    minimum_curvature = 150 # 150   ~500ft for highway
    if (middle_curverad_m < minimum_curvature):
        print("Curve < 150m")
        tracking_errors.curvature = True

    # check change of curvature of both lane lines independently
    change_factor = 3
    left_line_error = False
    right_line_error = False
    
    if (tracking_errors.first_frame == False and tracking_errors.curvature == False and tracking_errors.direction == False and tracking_errors.parallel == False and tracking_errors.distance == False):

        average_lane_distance = np.mean(last_frame.average_lane_distance, axis=0)
        if (left_curverad_m > change_factor*last_frame.left_line.last_radius_m or left_curverad_m < last_frame.left_line.last_radius_m/change_factor):
            print("Problem with left curvature")
            left_line_error = True
        if (right_curverad_m > change_factor*last_frame.right_line.last_radius_m or right_curverad_m < last_frame.right_line.last_radius_m/change_factor):
            print("Problem with right curvature")
            right_line_error = True
    else:
        average_lane_distance = np.mean(right_fitx - left_fitx)

    # Correct left or right line with a copy from the other "good" line
    # If both lines are bad use all data from last frame
    if (left_line_error == True and right_line_error == True):
        tracking_errors.curvature = True
    elif (left_line_error == True):
        print("Left line corrected")
        left_fitx = right_fitx - average_lane_distance
    elif (right_line_error == True):
        print("Right line corrected")
        right_fitx = left_fitx + average_lane_distance

    return


def process_image(img):

    # Save image size parameters
    img_size = (img.shape[1], img.shape[0])
    
    # Undistort image
    orig_img = cv2.undistort(img, mtx, dist, None, mtx)
    
    # Equalize the Y channel to correct changes in brightness
    img_yuv = cv2.cvtColor(orig_img, cv2.COLOR_BGR2YUV)
    # Create a CLAHE object
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # Apply Contrast Limited Adaptive Histogram Equalization
    img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
    # Convert the YUV image back to BGR format
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    # Detect lines in different color spaces
    hls_binary = hls_threshold(img, sthresh=(190,255)) # sthresh=(100,255)
    rgb_binary = rgb_threshold(img, rthresh=(247,255)) # rthresh=(200,255)

    # Calculate x and y sobel
    gradx = abs_sobel_thresh(img, orient='x', thresh=(25,255)) #12
    grady = abs_sobel_thresh(img, orient='y', thresh=(35,255)) #25

    # Combine x and y sobel images
    sobel = np.zeros_like(img[:,:,0]) 
    sobel[((gradx == 1) & (grady == 1))] = 1

    #sobel = dir_threshold(img, sobel_kernel=15, thresh=(0.7, 1.3))

    # Combine sobel results with color space results into a single image 
    preprocessImage = np.zeros_like(img[:,:,0])   
    preprocessImage[((sobel == 1) | (hls_binary == 1) | (rgb_binary == 1))] = 255

    
    """
    # Define trapezoidal on image for image transformation
    bot_width = .76 # percent of bottom trapezoid height 0.76
    mid_width = .08 # percent of middle trapezoid height 0.08
    height_pct = .62 # percent of trapezoid height 0.62
    bottom_trim = .92 # percent from top to bottom to avoid car hood 0.935

    src = np.float32([[img_size[0]*(.5-mid_width/2), img_size[1]*height_pct], [img_size[0]*(.5+mid_width/2), img_size[1]*height_pct], [img_size[0]*(.5+bot_width/2), img_size[1]*bottom_trim], [img_size[0]*(.5-bot_width/2), img_size[1]*bottom_trim]])
    offset = img_size[0]*.25
    dst = np.float32([[offset, 0], [img_size[0]-offset, 0], [img_size[0]-offset, img_size[1]], [offset, img_size[1]]])
    """

    # TRANSFORM THE IMAGE

    # Define source and destination trapezoidal on image for image transformation
    src = np.float32([[265,669], [577,460], [704,460], [1040,669]]) # TRACK 1
    #src = np.float32([[310,686], [563,514], [772,514], [1046,686]]) # short: [552,516], [785,516]  / middle: [531,492], [755,492] / long: [577,460], [704,460] / very long: [599,445], [680,445] // [265,669], [1040,669]
    dst = np.float32([[310,719], [310,0], [920,0], [920,719]])

    # perform the transform
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    binary_warped = cv2.warpPerspective(preprocessImage, M, img_size, flags=cv2.INTER_LINEAR)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255


    # FIND LANE LINE POINTS AND FIT POLYNOMIALS TO IT
    
    # For first frame at the beginning and after line detection was lost
    if (tracking_errors.first_frame == True or tracking_errors.lost_tracking == True):
        # Use sliding window approach
        left_fitx, right_fitx, ploty, out_img = sliding_window(binary_warped, out_img)
    # For every other frame 
    else:
        # Use margin-around-last-known-location-approach
        left_fitx, right_fitx, ploty, out_img = search_with_margin(binary_warped, out_img)


    # CALCULATE CURVATURES 

    # Calculate radius, direction and points for middle line
    left_curverad_m, middle_curverad_m, right_curverad_m, direction, middle_pts = calculate_middle_line(left_fitx, right_fitx, ploty)

    # Calculate position of the car in respect to the lane lines (calculated middle line)
    car_position = round((img_size[0]/2 - middle_pts[0, len(ploty)-1, 0]) * xm_per_pix, 2)

    # Calculate lane line distance for current frame
    average_lane_distance = np.mean(right_fitx - left_fitx)
    

    # SANITY CHECKS

    # Do not perform sanity checks on first frame
    if (tracking_errors.first_frame == True):

        # Smoothen the output by averaging right_fitx and left_fitx over a couple frames
        # First time fill list with copies of the first frame
        for i in range(0, last_frame.buffer_size):
            last_frame.average_left_fitx.append(left_fitx)
            last_frame.average_right_fitx.append(right_fitx)
            last_frame.average_lane_distance.append(average_lane_distance)

        # Save validated (and possibly corrected) values to last frame class variables
        last_frame.left_line.recent_fitx = left_fitx
        last_frame.left_line.last_radius_m = left_curverad_m
        last_frame.right_line.recent_fitx = right_fitx
        last_frame.right_line.last_radius_m = right_curverad_m
        last_frame.recent_direction = direction
        last_frame.recent_middle_pts = middle_pts
        last_frame.recent_radius = middle_curverad_m
        
        # Set first frame flag to False
        tracking_errors.first_frame = False

    # Not the first frame
    else:

        # Only perform sanity checks when tracking was not lost the frame before
        if (tracking_errors.lost_tracking == True):
            # Reset tracking error flag for next frame
            tracking_errors.lost_tracking = False
        else:
            # Calculate average distance from last buffered values
            average_lane_distance = np.mean(last_frame.average_lane_distance, axis=0)
            # Verify that lane lines have the correct distance, are parallel, and don't change directions abruptly
            check_lane_lines(left_fitx, right_fitx, direction, average_lane_distance)
            # Verify that lane lines have a certain curvature and check both lines independently
            check_line_curvature(left_curverad_m, middle_curverad_m, right_curverad_m, left_fitx, right_fitx)
   
    
            # Check if at least one sanity check for both lines has failed
            if ((tracking_errors.distance == True or tracking_errors.direction == True or tracking_errors.curvature == True or tracking_errors.parallel == True) and tracking_errors.lost_tracking == False):
                # add error description to error list
                tracking_errors.add_errors_to_list()
                # increase error counter by one
                tracking_errors.counter = tracking_errors.counter + 1
                # Reset error flags for next frame
                tracking_errors.distance = False
                tracking_errors.direction = False
                tracking_errors.curvature = False
                tracking_errors.parallel = False
                
                # Load values from last frame into variables
                left_fitx = last_frame.left_line.recent_fitx
                left_curverad_m = last_frame.left_line.last_radius_m
                right_fitx = last_frame.right_line.recent_fitx
                right_curverad_m = last_frame.right_line.last_radius_m
                direction = last_frame.recent_direction
                middle_pts = last_frame.recent_middle_pts
                middle_curverad_m = last_frame.recent_radius
                average_lane_distance = last_frame.average_lane_distance[last_frame.buffer_size-1]
                print("Use last frame.")

            # All results from sanity checks are okay
            else:
                # Reset error counter and empty error list
                tracking_errors.counter = 0
                del tracking_errors.error_list[:]
                
                
        # Save validated (and possibly corrected) values to last frame class variables
        last_frame.left_line.recent_fitx = left_fitx
        last_frame.left_line.last_radius_m = left_curverad_m
        last_frame.right_line.recent_fitx = right_fitx
        last_frame.right_line.last_radius_m = right_curverad_m
        last_frame.recent_direction = direction
        last_frame.recent_middle_pts = middle_pts
        last_frame.recent_radius = middle_curverad_m

        # add new values to the buffer and push out the oldest ones 
        last_frame.average_left_fitx.append(last_frame.left_line.recent_fitx)
        last_frame.average_right_fitx.append(last_frame.right_line.recent_fitx)
        last_frame.average_lane_distance.append(average_lane_distance)


    # Calculate the mean over all buffered values and save as new poynomial x values
    left_fitx = np.mean(last_frame.average_left_fitx, axis=0)
    right_fitx = np.mean(last_frame.average_right_fitx, axis=0)

    # Calculate (smoothened) radius, direction and points for middle line
    left_curverad_m, middle_curverad_m, right_curverad_m, direction, middle_pts = calculate_middle_line(left_fitx, right_fitx, ploty)

    # check if at least one sanity check has failed in all of the last {max_errors} frames
    if (tracking_errors.counter >= tracking_errors.max_errors):
        # print the description of the occured errors
        print (tracking_errors.error_list)
        # Start with the sliding window approach
        tracking_errors.lost_tracking = True
        # reset error counter and empty error list
        tracking_errors.counter = 0
        del tracking_errors.error_list[:]


    # generate polygons to show the lane and the lane lines
    line_marker_width = 20
    
    # creat points left and right of the line poly points 
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-line_marker_width, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+line_marker_width, ploty])))])    
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-line_marker_width, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+line_marker_width, ploty])))])
    # stack them together to form three different polygons
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
    lane_poly_pts = np.hstack((left_line_window2, right_line_window1))

    # create a new image for drawing
    marker_image = np.zeros_like(np.dstack((binary_warped, binary_warped, binary_warped))*255)
    # draw red and blue polys over lane lines
    cv2.fillPoly(marker_image, np.int_([left_line_pts]), (255,0, 0))
    cv2.fillPoly(marker_image, np.int_([right_line_pts]), (0,0, 255))
    # draw green poly over lane
    cv2.fillPoly(marker_image, np.int_([lane_poly_pts]), (0,120, 0))
    # draw white line in the middle
    cv2.polylines(marker_image, np.int_([middle_pts]), isClosed=False, color=(255,255,255), thickness=5, lineType=8, shift=0)
        
    # Warp the picture back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(marker_image, Minv, (img_size[0], img_size[1])) 
    # Combine the result with the original image
    result = cv2.addWeighted(orig_img, 1, newwarp, 0.8, 0)

    # Add text with error info to the image 
    cv2.putText(result, str(tracking_errors.counter) + ' errors', (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Add text with car position to the image 
    cv2.putText(result, 'Position: ' + str(car_position) + ' m', (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Add text with direction and radius info to the image 
    cv2.putText(result, direction + ', radius = ' + str(round(middle_curverad_m, 1)) + '(m)', (50, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    

    # create image in image video for tuning / debugging
    gradx[(gradx == 1)] = 255
    grady[(grady == 1)] = 255
    hls_binary[(hls_binary == 1)] = 255
    rgb_binary[(rgb_binary == 1)] = 255
    sobel_255 = np.zeros_like(img[:,:,0])
    sobel_255[((gradx == 255) & (grady == 255))] = 255
    c_binary = np.zeros_like(img[:,:,0])
    c_binary[((hls_binary == 255) | (rgb_binary == 255))] = 255
    
    first_pic = cv2.cvtColor(sobel_255, cv2.COLOR_GRAY2BGR) # sobel_255
    second_pic = cv2.cvtColor(c_binary, cv2.COLOR_GRAY2BGR) # hls_binary
    third_pic = cv2.cvtColor(preprocessImage, cv2.COLOR_GRAY2BGR) # preprocessImage
    fourth_pic = cv2.cvtColor(binary_warped, cv2.COLOR_GRAY2BGR) # out_img
    fifth_pic = out_img
    sixth_pic = marker_image #marker_image

    # Save Test Images
    index = 0
    write_name = './warped image/warped' + str(index) + '.jpg'
    cv2.imwrite(write_name, first_pic)
    index += 1
    write_name = './warped image/warped' + str(index) + '.jpg'
    cv2.imwrite(write_name, second_pic)
    index += 1
    write_name = './warped image/warped' + str(index) + '.jpg'
    cv2.imwrite(write_name, third_pic)
    index += 1
    write_name = './warped image/warped' + str(index) + '.jpg'
    cv2.imwrite(write_name, fourth_pic)
    index += 1
    write_name = './warped image/warped' + str(index) + '.jpg'
    cv2.imwrite(write_name, fifth_pic)
    index += 1
    write_name = './warped image/warped' + str(index) + '.jpg'
    cv2.imwrite(write_name, sixth_pic)
    index += 1
    

    first_pic = cv2.resize(first_pic, (0,0), fx=0.25, fy=0.25)
    second_pic = cv2.resize(second_pic, (0,0), fx=0.25, fy=0.25)
    third_pic = cv2.resize(third_pic, (0,0), fx=0.25, fy=0.25)
    fourth_pic = cv2.resize(fourth_pic, (0,0), fx=0.25, fy=0.25)
    fifth_pic = cv2.resize(fifth_pic, (0,0), fx=0.25, fy=0.25)
    sixth_pic = cv2.resize(sixth_pic, (0,0), fx=0.25, fy=0.25) 

    y_offset = 20
    
    # first row (pics 1-3)
    x_offset = result.shape[1]-first_pic.shape[1]-3*y_offset-second_pic.shape[1]-third_pic.shape[1]
    result[y_offset:y_offset+first_pic.shape[0], x_offset:x_offset+first_pic.shape[1]] = first_pic

    x_offset = result.shape[1]-first_pic.shape[1]-2*y_offset-third_pic.shape[1]
    result[y_offset:y_offset+second_pic.shape[0], x_offset:x_offset+second_pic.shape[1]] = second_pic

    x_offset = result.shape[1]-first_pic.shape[1]-y_offset
    result[y_offset:y_offset+third_pic.shape[0], x_offset:x_offset+third_pic.shape[1]] = third_pic
    
    # second row (pics 4-6)
    x_offset = result.shape[1]-first_pic.shape[1]-3*y_offset-second_pic.shape[1]-third_pic.shape[1]
    result[2*y_offset+first_pic.shape[0]:2*y_offset+first_pic.shape[0]+fourth_pic.shape[0], x_offset:x_offset+first_pic.shape[1]] = fourth_pic
    
    x_offset = result.shape[1]-first_pic.shape[1]-2*y_offset-third_pic.shape[1]
    result[2*y_offset+second_pic.shape[0]:2*y_offset+second_pic.shape[0]+fifth_pic.shape[0], x_offset:x_offset+fifth_pic.shape[1]] = fifth_pic
    
    x_offset = result.shape[1]-first_pic.shape[1]-y_offset
    result[2*y_offset+third_pic.shape[0]:2*y_offset+third_pic.shape[0]+sixth_pic.shape[0], x_offset:x_offset+sixth_pic.shape[1]] = sixth_pic


    return result



#Input_video = 'test.mp4'
Input_video = 'project_video.mp4'
#Input_video = 'challenge_video.mp4'

Output_video = 'output_video_highway.mp4'

last_frame = Last_Frame()
tracking_errors = Error()


clip1 = VideoFileClip(Input_video)
video_clip = clip1.fl_image(process_image)
video_clip.write_videofile(Output_video, audio=False)
                   

                   
