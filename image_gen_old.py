import cv2
import numpy as np
import glob
import pickle
from tracker import tracker

import matplotlib.pyplot as plt

# read in the saved objpoints and imgpoints
dist_pickle = pickle.load(open("./calibration_file/calibration_pickle.p", "rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

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

# make a list of test images
images = glob.glob('./test_images/test*.jpg')

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
    
    #plt.imshow(preprocessImage, cmap = plt.get_cmap('gray'))
    #plt.show()

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

    plt.imshow(binary_warped, cmap = plt.get_cmap('gray'))
    plt.show()

    #histogram = np.sum(warped[warped.shape[0]//2:,:], axis=0)
    #plt.plot(histogram)
    #plt.show()

    # INSERTED HERE:
    
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
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
    minpix = 100 #50
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

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)

    plt.show()


    # Assume you now have a new warped binary image 
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
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

    """
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)

    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))

    #cv2.polylines(window_img, example, isClosed=False, color=(0,255,0), thickness=5, lineType=8, shift=0)
    warped_color = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    plt.imshow(warped_color)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)

    plt.show()
    """

    line_marker_width = 20

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-line_marker_width, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+line_marker_width, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-line_marker_width, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+line_marker_width, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    lane_poly_pts = np.hstack((left_line_window2, right_line_window1))

    marker_image = np.zeros_like(np.dstack((binary_warped, binary_warped, binary_warped))*255)
    cv2.fillPoly(marker_image, np.int_([left_line_pts]), (255,0, 0))
    cv2.fillPoly(marker_image, np.int_([right_line_pts]), (0,0, 255))
    cv2.fillPoly(marker_image, np.int_([lane_poly_pts]), (0,120, 0))


    # Calculate line in the middle
    bottom_center = left_fitx[-1] + (right_fitx[-1] - left_fitx[-1])/2
    lower_third = left_fitx[len(left_fitx)/3] + (right_fitx[len(right_fitx)/3] - left_fitx[len(left_fitx)/3])/2
    upper_third = left_fitx[2*len(left_fitx)/3] + (right_fitx[2*len(right_fitx)/3] - left_fitx[2*len(left_fitx)/3])/2
    top_center = left_fitx[0] + (right_fitx[0] - left_fitx[0])/2
    center_pts_x = [bottom_center, lower_third, upper_third, top_center]
    center_pts_y = [ploty[-1], ploty[len(ploty)/3], ploty[2*len(ploty)/3], ploty[0]]

    middle_fit = np.polyfit(center_pts_y, center_pts_x, 2)
    middle_fitx = middle_fit[0]*ploty**2 + middle_fit[1]*ploty + middle_fit[2]
    middle_points = np.array([np.transpose(np.vstack([middle_fitx, ploty]))])

    cv2.polylines(marker_image, np.int_([middle_points]), isClosed=False, color=(255,255,255), thickness=5, lineType=8, shift=0)

    plt.imshow(marker_image)
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.show()

    # Right or left turn
    if middle_fitx[0] > middle_fitx[-1]:
        direction = "right"
    else:
        direction = "left"

    # Calculate curve radius
    y_eval = np.max(ploty)-200
    middle_curverad = ((1 + (2*middle_fit[0]*y_eval + middle_fit[1])**2)**1.5) / np.absolute(2*middle_fit[0])

    print (direction)
    print(middle_curverad)

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/500 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    middle_fit_real_world = np.polyfit(ploty*ym_per_pix, middle_fitx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    middle_curverad_m = ((1 + (2*middle_fit_real_world[0]*y_eval*ym_per_pix + middle_fit_real_world[1])**2)**1.5) / np.absolute(2*middle_fit_real_world[0])

    print(middle_curverad_m, 'm')


    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(marker_image, Minv, (img_size[0], img_size[1])) 

    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.8, 0)




    cv2.putText(result, direction + ', radius = ' + str(round(middle_curverad_m, 1)) + '(m)', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    """
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    print(left_curverad, 'm', right_curverad, 'm')
    """


    """
    # set up the overall class to do tracking
    window_width = 25
    window_height = 80
    curve_centers = tracker(Mywindow_width = window_width, Mywindow_height = window_height, Mymargin = 25, My_ym = 10/720, My_xm = 4/384, Mysmooth_factor = 15)

    window_centroids = curve_centers.find_window_centroids(binary_warped)

    # points used to draw all the left and right windows
    l_points = np.zeros_like(binary_warped)
    r_points = np.zeros_like(binary_warped)

    # points used to find the left and right lanes
    rightx = []
    leftx = []

    # go through each level and draw windows
    for level in range(0, len(window_centroids)):
        # window_mask is a function to draw window areas
        l_mask = window_mask(window_width, window_height, binary_warped, window_centroids[level][0], level)
        r_mask = window_mask(window_width, window_height, binary_warped, window_centroids[level][1], level)
        # add center value found in frame to the list of lane points per left, right
        leftx.append(window_centroids[level][0])
        rightx.append(window_centroids[level][1])
        # add graphic points from window mask here to total pixels found
        l_points[(l_points == 255) | ((l_mask == 1))] = 255
        r_points[(r_points == 255) | ((r_mask == 1))] = 255

    
    # draw the results
    template = np.array(r_points + l_points, np.uint8) # add both left and right window pixels together
    zero_channel = np.zeros_like(template) # create a zero color channel
    template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8) # make window pixels green
    warpage = np.array(cv2.merge((binary_warped, binary_warped, binary_warped)), np.uint8) # making the original road pixels 3 color channels
    result = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the original road image with window results
    

    plt.imshow(result, cmap = plt.get_cmap('gray'))
    plt.show()


    # fit the lane boundaries to the left, right center position found
    yvals = range(0, binary_warped.shape[0])

    res_yvals = np.arange(binary_warped.shape[0] - (window_height/2), 0, -window_height)
    
    left_fit = np.polyfit(res_yvals, leftx, 2)
    
    left_fitx = left_fit[0] * yvals * yvals + left_fit[1] * yvals + left_fit[2]
    left_fitx = np.array(left_fitx, np.int32)

    right_fit = np.polyfit(res_yvals, rightx, 2)
    right_fitx = right_fit[0] * yvals * yvals + right_fit[1] * yvals + right_fit[2]
    right_fitx = np.array(right_fitx, np.int32)

    left_lane = np.array(list(zip(np.concatenate((left_fitx - window_width/2, left_fitx[::-1] + window_width/2), axis=0), np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)
    right_lane = np.array(list(zip(np.concatenate((right_fitx - window_width/2, right_fitx[::-1] + window_width/2), axis=0), np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)
    middle_marker = np.array(list(zip(np.concatenate((right_fitx - window_width/2, right_fitx[::-1] + window_width/2), axis=0), np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)

    road = np.zeros_like(img)
    road_bkg = np.zeros_like(img)
    cv2.fillPoly(road, [left_lane], color=[255, 0, 0])
    cv2.fillPoly(road, [right_lane], color=[0, 0, 255])
    cv2.fillPoly(road_bkg, [left_lane], color=[255, 255, 255])
    cv2.fillPoly(road_bkg, [right_lane], color=[255, 255, 255])

    road_warped = cv2.warpPerspective(road, Minv, img_size, flags=cv2.INTER_LINEAR)
    road_warped_bkg = cv2.warpPerspective(road_bkg, Minv, img_size, flags=cv2.INTER_LINEAR)

    #result = road
    base = cv2.addWeighted(img, 1.0, road_warped_bkg, -1.0, 0.0)
    result = cv2.addWeighted(base, 1.0, road_warped, 1.0, 0.0)

    # Grab values from curve centers
    xm_per_pix = curve_centers.xm_per_pix
    ym_per_pix = curve_centers.ym_per_pix

    curve_fit_cr = np.polyfit(np.array(res_yvals, np.float32) * ym_per_pix, np.array(leftx, np.float32) * xm_per_pix, 2)
    curverad = ((1 + (2 * curve_fit_cr[0] * yvals[-1] * ym_per_pix + curve_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * curve_fit_cr[0])

    # calculate the offset of the car on the road
    camera_center = (left_fitx[-1] + right_fitx[-1])/2
    center_diff = (camera_center - binary_warped.shape[1]/2) * xm_per_pix
    side_pos = 'left'
    if center_diff <= 0:
        side_pos = 'right'

    # draw the text showing curvature, offset, and speed
    cv2.putText(result, 'Radius of Curvature = ' + str(round(curverad, 3)) + '(m)', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(result, 'Vehicle is ' + str(abs(round(center_diff, 3))) + 'm ' + side_pos + ' of center', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    """



    write_name = './output_images/tracked' + str(idx) + '.jpg'
    cv2.imwrite(write_name, result)
                   

                   
