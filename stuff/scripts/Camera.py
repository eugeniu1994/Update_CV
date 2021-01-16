#!/usr/bin/env python2.7

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

nRows, nCols = 9, 6
dimension = 22  # - mm

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, dimension, 0.001)

def calibCam(img):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((nRows * nCols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nRows, 0:nCols].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    #images = [img]#glob.glob(r'images/*.jpg')
    images = glob.glob(img+'/*.png')
    print('images path ', images)
    found = 0
    for fname in images:  # Here, 10 can be changed to whatever number you like to choose
        print('fname ',fname)
        img = cv2.imread(fname)  # Capture frame-by-frame
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (nRows, nCols), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)  # Certainly, every loop objp is the same, in 3D.
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (nRows, nCols), corners2, ret)
            found += 1
            cv2.imshow('img', img)
            cv2.waitKey(1)
            '''while True:
                k = cv2.waitKey(1)
                if k == 27:
                    cv2.destroyAllWindows()
                    break'''

    print("Number of images used for calibration: ", found)
    cv2.destroyAllWindows()

    # calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print('mtx ',mtx)
    print('dist ', dist)
    print('rvecs ', np.shape(rvecs))
    print('tvecs ', np.shape(tvecs))
    np.save('mtx',mtx)
    np.save('dist', dist)
    np.save('rvecs', rvecs)
    np.save('tvecs', tvecs)

def testCalibration(file, lidar):
    def draw(img, corners, imgpts):
        corner = tuple(corners[0].ravel())
        img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
        img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
        img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
        return img

    objp = np.zeros((nRows * nCols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nRows, 0:nCols].T.reshape(-1, 2)

    axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)

    mtx = np.load('mtx.npy')
    dist = np.load('dist.npy')
    tvecs_ = np.load('tvecs.npy')
    print('tvecs_ ', np.shape(tvecs_))

    print('K')
    print(mtx)
    img = cv2.imread(file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (nRows, nCols), None)
    print('ret ',ret)

    skip = 2
    lidarPoints = np.load(lidar)[1::skip, :]
    mean = np.mean(lidarPoints, axis=0)
    print('mean ',mean)
    #lidarPoints = lidarPoints-mean
    print('lidarPoints ',np.shape(lidarPoints))
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection="3d")

    #ax1.set_axis_off()
    #ax1.set_facecolor((0, 0, 0))
    ax1.set_xlabel('X', fontsize=10)
    ax1.set_ylabel('Y', fontsize=10)
    ax1.set_zlabel('Z', fontsize=10)

    #ax1.scatter(lidarPoints[:,0],lidarPoints[:,1],lidarPoints[:,2], s=0.3)
    #ax1.set_aspect("equal")
    #ax1.set_aspect('equal', 'box')
    l=6
    #ax1.set_xlim3d(0, l)
    #ax1.set_ylim3d(-3, 3)
    #ax1.set_zlim3d(0, 3)

    if ret == True:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        ret, rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, dist)
        #axis = np.float32(lidarPoints[1::5, :]).reshape(-1, 3)

        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
        img = draw(img, corners2, imgpts)
        img = cv2.resize(img, (960, 540))
        #cv2.imshow('Image ', img)

        fig = plt.figure()
        plt.imshow(img)
        '''while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break'''

    print('rvecs:{}, tvecs:{}'.format(np.shape(rvecs), np.shape(tvecs)))

    inverse_camera_matrix_new = np.linalg.inv(mtx)
    Translation = np.asarray(tvecs).squeeze()

    R_mtx, jac = cv2.Rodrigues(rvecs)
    inverse_R_mtx = np.linalg.inv(R_mtx)

    def compute_XYZ(u, v, s=1):  # from 2D pixels to 3D world
        uv_ = np.array([[u, v, 1]], dtype=np.float32).T
        suv_ = s * uv_
        xyz_ = inverse_camera_matrix_new.dot(suv_) - Translation
        XYZ = inverse_R_mtx.dot(xyz_)

        pred = XYZ.T[0]
        #print('p is ',pred)
        return pred

    print('corners2:{}'.format(np.shape(corners2)))
    print('corners2[0].ravel():{}'.format(np.shape(corners2[0].ravel())))
    corner = tuple(corners2[0].ravel())

    plane= []
    for c in corners2:
        corn = np.asarray(c).squeeze()
        p = compute_XYZ(corn[0], corn[1])
        #p -= mean
        ax1.scatter(p[0], p[1], p[2], s=1, c='r')
        plane.append(p)

    plane = np.array(plane)

    def PCA(data, correlation=False, sort=True):
        # data = nx3
        mean = np.mean(data, axis=0)
        data_adjust = data - mean
        #: the data is transposed due to np.cov/corrcoef syntax
        if correlation:
            matrix = np.corrcoef(data_adjust.T)
        else:
            matrix = np.cov(data_adjust.T)
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        if sort:
            #: sort eigenvalues and eigenvectors
            sort = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[sort]
            eigenvectors = eigenvectors[:, sort]

        return eigenvalues, eigenvectors

    eigenvalues, eigenvectors = PCA(plane)
    w, v = PCA(plane)
    point = np.mean(plane, axis=0)
    w *= 1
    p3 = ax1.quiver([point[0]], [point[1]], [point[2]], [v[0, :] * np.sqrt(w[0])], [v[1, :] * np.sqrt(w[0])],
                   [v[2, :] * np.sqrt(w[0])], linewidths=(1.8,))
    print('w ',w)
    print('v[0] ' ,v[0,:])


    t = np.asarray([2.5,0,.5]).squeeze()
    print('t is ', t)
    #p[-1] = 0
    #p = compute_XYZ(corner[0], corner[1])
    #ax1.scatter(t[0], t[0], t[0], s=5, c='g')
    #ax1.scatter(p[0], p[1], p[2], s=10, c='g')

    plt.show()


if __name__ == '__main__':
    fname = '../DATA/img/snapshot_640_480_2.jpg'
    fname = '/home/eugen/catkin_ws/src/Camera_Lidar/DATA/data_img'
    #calibCam(fname)

    imgtest = '/home/eugen/catkin_ws/src/Camera_Lidar/DATA/data_img/img_1610553723407007932.png'
    lidar = '/home/eugen/catkin_ws/src/Camera_Lidar/DATA/data_pointCloud/cloud_points_1610553723407007932.npy'
    #testCalibration(imgtest,lidar)

    shape = (1, 4, 3)
    source = np.zeros(shape, np.float32)

    # [x, y, z]
    source[0][0] = [857, 120, 854]
    source[0][1] = [254, 120, 855]
    source[0][2] = [256, 120, 255]
    source[0][3] = [858, 120, 255]

    target = source * 10

    retval, M, inliers = cv2.estimateAffine3D(source, target)
    print('retval ',retval)
    print('M ', M)
    print('inliers ', inliers)