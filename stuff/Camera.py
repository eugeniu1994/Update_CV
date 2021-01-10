#!/usr/bin/env python2.7

import numpy as np
import cv2

def calibCam(img):
    import numpy as np
    import cv2
    import glob
    import yaml

    dimension = 22  # - mm
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, dimension, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((9 * 6, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    images = [img]#glob.glob(r'images/*.jpg')

    found = 0
    for fname in images:  # Here, 10 can be changed to whatever number you like to choose
        print('fname ',fname)
        img = cv2.imread(fname)  # Capture frame-by-frame
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)  # Certainly, every loop objp is the same, in 3D.
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (9, 6), corners2, ret)
            found += 1
            cv2.imshow('img', img)
            while True:
                k = cv2.waitKey(1)
                if k == 27:
                    cv2.destroyAllWindows()
                    break

    print("Number of images used for calibration: ", found)
    cv2.destroyAllWindows()

    # calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print('Final objpoints ',objpoints)
    print('Final imgpoints ',imgpoints)
    # transform the matrix and distortion coefficients to writable lists
    data = {'camera_matrix': np.asarray(mtx).tolist(),
            'dist_coeff': np.asarray(dist).tolist()}
    print('K ',mtx)
    print('dist ',dist)
    print('rvecs ', np.shape(rvecs))
    print('tvecs ', np.shape(tvecs))
    # and save it to a file
    #with open("calibration_matrix.yaml", "w") as f:
        #yaml.dump(data, f)

if __name__ == '__main__':
    fname = '../DATA/img/snapshot_640_480_2.jpg'
    calibCam(fname)

    '''nRows, nCols = 9, 6
    dimension = 22  # - mm
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, dimension, 0.001)
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("Reading ", fname)
    ret, corners = cv2.findChessboardCorners(gray, (nCols, nRows), None)
    if ret == True:
        print("ESC to skip or ENTER to accept")
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(img, (nCols, nRows), corners2, ret)
        cv2.imshow('img', img)
        while True:
            k = cv2.waitKey(1)
            if k == 27:
                cv2.destroyAllWindows()
                break
    else:
        print('cannot detect chessboard')'''