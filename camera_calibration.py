import cv2
import numpy as np
import glob
import pickle


# prepare object points (0,0,0), (1,0,0),...(8,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

# arrays to store object points and image points from all images
objpoints = [] #3D points in real world
imgpoints = [] #2D points in image plane

# list of calibration images
images = glob.glob('./camera_cal/calibration*.jpg')

#print(images)

# go through the list and search for chessboard corners
for idx, fname in enumerate(images):
    # read image and convert to gray
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # find corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

    # if found, add object points and image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # draw and display corners
        cv2.drawChessboardCorners(img, (9,6), corners, ret)
        write_name = './corners_found/corners_found' + str(idx) + '.jpg'
        cv2.imwrite(write_name, img)

# load image for reference
img = cv2.imread('./camera_cal/calibration1.jpg')
img_size = (img.shape[1], img.shape[0])

print(imgpoints)
    
# do camera calibration with object and image points
ret, mtx, dist, rvecs, tvecs =  cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

# save camera calibration result
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump(dist_pickle, open("./calibration_file/calibration_pickle.p", "wb")) 

