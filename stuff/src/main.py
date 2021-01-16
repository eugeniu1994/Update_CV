#!/usr/bin/env python2.7

'''
author: Eugeniu Vezeteu
Camera & Lidar subscriber node
subscribe to camera & lidar topics
get image & pointcloud
display them for later calibration
save img and pointcloud in files
'''
from __future__ import print_function

import rospy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
import message_filters
import image_geometry
import ros_numpy

import cv2
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
import os

import pcl

CV_BRIDGE = CvBridge()


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


def getPointCoud(points, ax, plotInit=False, plotSmoothed=False, plotFinal=True):
    points = np.asarray(points, dtype=np.float32)
    p = pcl.PointCloud(points)

    # smoothing
    fil = p.make_statistical_outlier_filter()
    fil.set_mean_k(50)
    fil.set_std_dev_mul_thresh(1.0)
    smoothed = np.array(fil.filter())

    def do_ransac_plane_segmentation(pcl_data, pcl_sac_model_plane, pcl_sac_ransac, max_distance):
        """
        Create the segmentation object
        :param pcl_data: point could data subscriber
        :param pcl_sac_model_plane: use to determine plane models
        :param pcl_sac_ransac: RANdom SAmple Consensus
        :param max_distance: Max distance for apoint to be considered fitting the model
        :return: segmentation object
        """
        seg = pcl_data.make_segmenter()
        seg.set_model_type(pcl_sac_model_plane)
        seg.set_method_type(pcl_sac_ransac)
        seg.set_distance_threshold(max_distance)

        inliers, coefficients = seg.segment()
        inlier_object = pcl_data.extract(inliers, negative=False)
        outlier_object = pcl_data.extract(inliers, negative=True)
        return inlier_object, outlier_object

    # RANSAC Plane Segmentation
    points = np.asarray(smoothed, dtype=np.float32)
    p = pcl.PointCloud(points)

    inlier, outliner = do_ransac_plane_segmentation(p, pcl.SACMODEL_PLANE, pcl.SAC_RANSAC, 0.03)
    inlier, outliner = np.array(inlier), np.array(outliner)

    def do_ransac_plane_normal_segmentation(point_cloud, input_max_distance):
        segmenter = point_cloud.make_segmenter_normals(ksearch=50)
        segmenter.set_optimize_coefficients(True)
        segmenter.set_model_type(pcl.SACMODEL_NORMAL_PLANE)  # pcl_sac_model_plane
        segmenter.set_normal_distance_weight(0.1)
        segmenter.set_method_type(pcl.SAC_RANSAC)  # pcl_sac_ransac
        segmenter.set_max_iterations(100)
        segmenter.set_distance_threshold(input_max_distance)  # 0.03)  #max_distance
        indices, coefficients = segmenter.segment()

        print('Model coefficients: ' + str(coefficients[0]) + ' ' + str(
            coefficients[1]) + ' ' + str(coefficients[2]) + ' ' + str(coefficients[3]))

        print('Model inliers: ' + str(len(indices)))

        inliers = point_cloud.extract(indices, negative=False)
        outliers = point_cloud.extract(indices, negative=True)

        return coefficients, inliers, outliers

    # coefficients, inliers, outliers = do_ransac_plane_normal_segmentation(p, 0.01)
    # inlier, outliner = np.array(inliers), np.array(outliers)
    if plotFinal:
        p1 = ax.scatter(outliner[:, 0], outliner[:, 1], outliner[:, 2], c='r', s=0.2)
        p2 = ax.scatter(inlier[:, 0], inlier[:, 1], inlier[:, 2], c='g', s=0.2)

        w, v = PCA(inlier)
        point = np.mean(inlier, axis=0)
        w *= 3
        p3 = ax.quiver([point[0]], [point[1]], [point[2]], [v[0, :] * np.sqrt(w[0])], [v[1, :] * np.sqrt(w[0])],
                       [v[2, :] * np.sqrt(w[0])], linewidths=(1.8,))

        return [p1, p2, p3], inlier


def callback(image, lidar):
    global cloud_disp, img_disp, cloud_points, images, idx
    cloud_points, images = [], []
    now = rospy.get_rostime()
    rospy.loginfo("receiving data, now:{}".format(now))
    idx = now
    def process_Images(image, ax, view_Corners = False):
        nRows, nCols = 9, 6
        dimension = 22  # - mm
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, dimension, 0.001)
        try:
            frame = CV_BRIDGE.imgmsg_to_cv2(image, "bgr8")
            #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #ret, corners = cv2.findChessboardCorners(gray, (nCols, nRows), None)
            #if ret == True:
            #    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            #    cv2.drawChessboardCorners(frame, (nCols, nRows), corners2, ret)

        except CvBridgeError, e:
            rospy.logerr(e)
        img_disp = np.array(frame, dtype=np.uint8)

        def showImageCV(img):
            resized = cv2.resize(img, None, fx=0.5, fy=0.5)
            cv2.imshow('image', resized)
            cv2.waitKey(1)

        # showImageCV(img)
        ax.cla()
        images = img_disp
        img_disp = ax.imshow(img_disp)
        ax.set_axis_off()

        return images

    def process_Points(lidar, ax):
        # rospy.loginfo(lidar)
        # Extract points data
        points = ros_numpy.point_cloud2.pointcloud2_to_array(lidar)
        points = np.asarray(points.tolist())

        #inrange = np.where((points[:, 0] > 0) & (points[:, 0] < 5) & (np.abs(points[:, 1]) < 1.5) & (points[:, 2] < 2))
        inrange = np.where((points[:, 0] > 0) & (points[:, 0] < 5.5) &
                           (np.abs(points[:, 1]) < 2.5) & (points[:, 2] < 2.5))

        points = points[inrange[0]]
        skip = 1
        points = points[1::skip, :3]
        rospy.loginfo('points {}'.format(np.shape(points)))

        # cmap = matplotlib.cm.get_cmap('hsv')
        # colors = cmap(points[:, -1] / np.max(points[:, -1]))

        ax.cla()
        #global cloud_disp
        # cloud_disp.remove()
        # cloud_disp = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=0.2)
        cloud_disp = ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.2)
        #cloud_disp, Cloud3D = getPointCoud(points=points, ax=ax, plotInit=False, plotSmoothed=False, plotFinal=True)

        ax.set_xlabel('X', fontsize=10)
        ax.set_ylabel('Y', fontsize=10)
        ax.set_zlabel('Z', fontsize=10)

        #ax.set_axis_off()
        return points

    #images = process_Images(image=image, ax=ax2)
    cloud_points = process_Points(lidar=lidar, ax=ax1)
    #saveData(cloud_points=cloud_points, images=images, index=idx)

    #save_d = os.path.join('/home/eugen/catkin_ws/src/Camera_Lidar/DATA/','data.csv')
    #np.savetxt(save_d, cloud_points, delimiter=',')
    #rospy.loginfo('Data saved: {}-------------------------'.format(np.shape(cloud_points)))

    #fig.canvas.draw_idle()
    fig.canvas.draw()

def controlGetData(camera, lidar):
    plt.ion()
    plt.show()
    rospy.loginfo('here')
    while not rospy.is_shutdown():
        camData = None#rospy.wait_for_message(camera, Image)
        lidarData = rospy.wait_for_message(lidar, PointCloud2)

        #process both here
        callback(image=camData, lidar=lidarData)
        rospy.loginfo('Boom')

        #plt.pause(.5)

    rospy.spin()

def saveData(cloud_points=None, images=None, index=0):
    data_pointCloud = os.path.join('/home/eugen/catkin_ws/src/Camera_Lidar/DATA/data_pointCloud','cloud_points_{}.npy'.format(index))
    data_img = os.path.join('/home/eugen/catkin_ws/src/Camera_Lidar/DATA/data_img', 'img_{}.png'.format(index))

    if cloud_points is not None:
        np.save(data_pointCloud, cloud_points)
    if images is not None:
        cv2.imwrite(data_img, images)

    rospy.loginfo('Data saved ,cloud_disp:{},images:{}'.format(np.shape(cloud_points),np.shape(images)))

if __name__ == '__main__':
    rospy.init_node('Camera_Lidar')
    rospy.loginfo('Camera_Lidar started----------------')

    global cloud_points, images, idx
    idx=0
    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122)
    #ax1.set_axis_off()
    #ax1.set_facecolor((0, 0, 0))
    ax1.set_xlabel('X', fontsize=10)
    ax1.set_ylabel('Y', fontsize=10)
    ax1.set_zlabel('Z', fontsize=10)
    #plt.subplots_adjust(left=0.05, bottom=0.2)
    global cloud_disp
    cloud_disp = ax1.scatter([], [], [])

    def update(val):
        rospy.logerr('Button clicked')
        # make global the current pointcloud and the image
        global cloud_points, images
        rospy.logerr('cloud_points:{}, images:{}'.format(np.shape(cloud_points), np.shape(images)))
        # save pointcloud and image on some folders
        saveData(cloud_points=cloud_points, images=images, index=idx)
        # fig.canvas.draw_idle()

    btnSave = Button(plt.axes([0.81, 0.05, 0.15, 0.075]), 'Save points', color='white')
    btnSave.on_clicked(update)

    # Subscribe to camera & lidar
    image_topic_name = rospy.get_param('~image_topic_name', '/pylon_camera_node/image_raw')
    lidar_topic_name = rospy.get_param('~lidar_topic_name', '/velodyne_points')

    mood = True
    if mood:
        controlGetData(camera=image_topic_name, lidar=lidar_topic_name)
    else:
        image_sub = message_filters.Subscriber(image_topic_name, Image)
        lidar_sub = message_filters.Subscriber(lidar_topic_name, PointCloud2)
        # Synchronize the topics by time
        time_syncronizer = message_filters.ApproximateTimeSynchronizer(
            [image_sub, lidar_sub], queue_size=5, slop=.1)

        #time_syncronizer = message_filters.TimeSynchronizer([image_sub, lidar_sub], 10)
        time_syncronizer.registerCallback(callback)
        plt.show()

    def Shutdown():
        rospy.loginfo('------------------Shutdown----------------------')
        plt.close()
        cv2.destroyAllWindows()

    rospy.on_shutdown(Shutdown)

    rospy.loginfo('spin')
    rospy.spin()
