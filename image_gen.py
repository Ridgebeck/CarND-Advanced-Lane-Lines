import cv2
import numpy as np
import glob
import pickle
from tracker import tracker
import collections

import matplotlib.pyplot as plt

# Define conversions in x and y from pixels space to meters
ym_per_pix = 45/600 # meters per pixel in y dimension
xm_per_pix = 4/475 # meters per pixel in x dimension

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
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        # average x values of the fitted line over the last n iterations
        self.bestx = None 
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        # radius of curvature of the line in some units
        self.radius_of_curvature = None 
        # distance in meters of vehicle center from the line
        self.line_base_pos = None 
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        # x values for detected line pixels
        self.allx = None  
        # y values for detected line pixels
        self.ally = None

class Last_Frame():
    def __init__(self):
        self.recent_radius = None
        self.recent_direction = "straight"
        self.recent_middle_pts = None
        self.buffer_size = 1 # 1 = no buffer
        self.first_frame = True
        self.average_left_fitx = collections.deque([], self.buffer_size)
        self.average_right_fitx = collections.deque([], self.buffer_size)
                
        self.left_line = Line()
        self.right_line = Line()

class Error():
    def __init__(self):
        # Reset tracking mode
        self.lost_tracking = True # Set to True for the first frame
        # Counter for counting frames with at least one error
        self.counter = 0
        # Error for not deteting lane lines in the last images
        self.no_lines = False # Set to True for the first frame
        # Error for curvature being out of range
        self.curvature = False
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
        if self.distance == True:
            error_message += "distance "
        if self.direction == True:
            error_message += "direction "
        self.error_list.append(error_message)

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0,255)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    # apply threshold
    binary_output[(scaled_sobel >= thresh[0])&(scaled_sobel <= thresh[1])] = 1
    return binary_output

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
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    with np.errstate(divide='ignore', invalid='ignore'):
        absgraddir = np.absolute(np.arctan(sobely/sobelx))
        binary_output = np.zeros_like(absgraddir)
        # apply threshold
        binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return binary_output
    
def color_threshold(image, sthresh=(0,255), vthresh=(0,255)):
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    s_channel = hls[:,:,2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >=sthresh[0]) & (s_channel <= sthresh[1])] = 1

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:,:,2]
    v_binary = np.zeros_like(v_channel)
    # apply threshold
    v_binary[(v_channel >=sthresh[0]) & (v_channel <= sthresh[1])] = 1

    output = np.zeros_like(s_channel)
    output[(s_binary == 1) & (v_binary == 1)] = 1
    return output    

def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height), max(0,int(center-width)):min(int(center+width), img_ref.shape[1])] = 1
    return output

def calculate_middle_line(left_fitx, right_fitx, ploty):
    # Calculate line in the middle
    bottom_center = left_fitx[-1] + (right_fitx[-1] - left_fitx[-1])/2
    lower_third = left_fitx[len(left_fitx)/3] + (right_fitx[len(right_fitx)/3] - left_fitx[len(left_fitx)/3])/2
    upper_third = left_fitx[2*len(left_fitx)/3] + (right_fitx[2*len(right_fitx)/3] - left_fitx[2*len(left_fitx)/3])/2
    top_center = left_fitx[0] + (right_fitx[0] - left_fitx[0])/2
    center_pts_x = [bottom_center, lower_third, upper_third, top_center]
    center_pts_y = [ploty[-1], ploty[len(ploty)/3], ploty[2*len(ploty)/3], ploty[0]]
    middle_fit = np.polyfit(center_pts_y, center_pts_x, 2)
    middle_fitx = middle_fit[0]*ploty**2 + middle_fit[1]*ploty + middle_fit[2]
    middle_pts = np.array([np.transpose(np.vstack([middle_fitx, ploty]))])

    # fit new polynomials to x,y in real world space
    middle_fit_real_world = np.polyfit(ploty*ym_per_pix, middle_fitx*xm_per_pix, 2)
    # calculate radius for a position 200 pixels up from the bottom frame
    y_eval = np.max(ploty)-200 
    # calculate radius for real world space (meters)
    middle_curverad_m = ((1 + (2*middle_fit_real_world[0]*y_eval*ym_per_pix + middle_fit_real_world[1])**2)**1.5) / np.absolute(2*middle_fit_real_world[0])
    # middle_curverad = ((1 + (2*middle_fit[0]*y_eval + middle_fit[1])**2)**1.5) / np.absolute(2*middle_fit[0]) # in radians

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
    
    return middle_curverad_m, direction, middle_pts


# make a list of test images
images = glob.glob('./test_images/test*.jpg')

last_frame = Last_Frame()
tracking_errors = Error()

for idx, fname in enumerate(images):
    # read and undistort image
    img = cv2.imread(fname)
    img = cv2.undistort(img, mtx, dist, None, mtx)

    # process image and generate binary pixels of interest
    preprocessImage = np.zeros_like(img[:,:,0])    
    gradx = abs_sobel_thresh(img, orient='x', thresh=(25,255)) #12
    grady = abs_sobel_thresh(img, orient='y', thresh=(35,255)) #25
    c_binary = color_threshold(img, sthresh=(100,255), vthresh=(50,255))
    preprocessImage[((gradx == 1) & (grady == 1) | (c_binary == 1))] = 255

    # work on defining perspective transformation area
    img_size = (img.shape[1], img.shape[0])
    bot_width = .76 # percent of bottom trapezoid height 0.76
    mid_width = .08 # percent of middle trapezoid height 0.08
    height_pct = .62 # percent of trapezoid height 0.62
    bottom_trim = .92 # percent from top to bottom to avoid car hood 0.935

    src = np.float32([[img_size[0]*(.5-mid_width/2), img_size[1]*height_pct], [img_size[0]*(.5+mid_width/2), img_size[1]*height_pct], [img_size[0]*(.5+bot_width/2), img_size[1]*bottom_trim], [img_size[0]*(.5-bot_width/2), img_size[1]*bottom_trim]])
    offset = img_size[0]*.25
    dst = np.float32([[offset, 0], [img_size[0]-offset, 0], [img_size[0]-offset, img_size[1]], [offset, img_size[1]]])

    #src = np.float32([[281,659], [598,446], [681,446], [1027,659]])
    #dst = np.float32([[381,659], [381,0], [927,0], [927,659]])

    # perform the transform
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    #warped = cv2.warpPerspective(preprocessImage, M, img_size, flags=cv2.INTER_LINEAR)
    binary_warped = cv2.warpPerspective(preprocessImage, M, img_size, flags=cv2.INTER_LINEAR)

    # For first frame at the beginning and after line detection was lost (sliding window approach)
    if tracking_errors.lost_tracking == True:
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
        # Create an output image to draw on and  visualize the result
        #out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
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
            #cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
            #cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
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

        # Set back error after first frame (beginning and after loss) was processed
        tracking_errors.lost_tracking = False

        # save polinomial fit from current frame
        last_frame.left_line.recent_fit = left_fit
        last_frame.right_line.recent_fit = right_fit
        
    else:
        
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
        margin = 100
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
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]


    # Calculate radius, direction and points for middle line
    middle_curverad_m, direction, middle_pts = calculate_middle_line(left_fitx, right_fitx, ploty)



    # SANITY CHECKS
    
    # detected lanes are too wide or too narrow
    lane_distance = np.mean(right_fitx - left_fitx)
    if ((lane_distance > (4.7 / xm_per_pix)) or (lane_distance < (3.3 / xm_per_pix))):
        tracking_errors.distance = True
    else:
        tracking_errors.distance = False
        
    # detected curve changes direction abruptly
    if (last_frame.recent_direction == "right" and direction == "left" or last_frame.recent_direction == "left" and direction == "right"):
        tracking_errors.direction  = True
        
    # curvature is to narrow
    minimum_curvature = 150 # ~500ft for highway
    if (middle_curverad_m < minimum_curvature):
        tracking_errors.curvature = True

    # Check if at least one sanity check has failed for the last three frames
    if (tracking_errors.distance == True """or tracking_errors.direction == True""" or tracking_errors.curvature == True or tracking_errors.lost_tracking == True):
        # add error description to error list
        tracking_errors.add_errors_to_list()
        # increase error counter by one
        tracking_errors.counter = tracking_errors.counter + 1
        
        # Use values from last frame
        left_fitx = last_frame.left_line.recent_fitx
        right_fitx = last_frame.right_line.recent_fitx
        middle_curverad_m = last_frame.recent_radius
        direction = last_frame.recent_direction
        middle_pts = last_frame.recent_middle_pts

        print("Use last frame.")
        
    else:
        # Reset error counter and empty error list
        tracking_errors.counter = 0
        del tracking_errors.error_list[:]
        
    # Save validated values to last frame class variables
    last_frame.left_line.recent_fitx = left_fitx
    last_frame.right_line.recent_fitx = right_fitx
    last_frame.recent_radius = middle_curverad_m
    last_frame.recent_direction = direction
    last_frame.recent_middle_pts = middle_pts

    # check if at least one sanity check has failed in the last three frames
    if (tracking_errors.counter >= 3):
        # print the description of the occured errors
        print (tracking_errors.error_list)
        # reset tracking method to histogram / sliding window approach for next frame
        tracking_errors.lost_tracking = True
        # reset error counter and empty error list
        tracking_errors.counter = 0
        del tracking_errors.error_list[:]
    

    # smoothen the output by averaging right_fitx and left_fitx over a couple pictures
    # first time fill list with copies of the first frame
    if last_frame.first_frame == True:
        for i in range(0, last_frame.buffer_size):
            last_frame.average_left_fitx.append(left_fitx)
            last_frame.average_right_fitx.append(right_fitx)
        last_frame.first_frame = False
    # only add new values and push out the oldest one 
    else:
        last_frame.average_left_fitx.append(left_fitx)
        last_frame.average_right_fitx.append(right_fitx)

    # Calculate the mean over all buffered values and save as new poynomial x values
    left_fitx = np.mean(last_frame.average_left_fitx, axis=0)
    right_fitx = np.mean(last_frame.average_right_fitx, axis=0)
    
    # Calculate (smoothened) radius, direction and points for middle line
    middle_curverad_m, direction, middle_pts = calculate_middle_line(left_fitx, right_fitx, ploty)


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
    result = cv2.addWeighted(img, 1, newwarp, 0.8, 0)
    # Add text with tracking info to the image 
    cv2.putText(result, direction + ', radius = ' + str(round(middle_curverad_m, 1)) + '(m)', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    write_name = './output_images/tracked' + str(idx) + '.jpg'
    cv2.imwrite(write_name, result)
                   

                   
