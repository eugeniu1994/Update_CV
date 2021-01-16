import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib
from mpl_toolkits.mplot3d import proj3d
import os

def extract_points_2D(fname):
    img = cv2.imread(fname)
    disp = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Select 2D Image Points')
    ax.set_axis_off()
    ax.imshow(disp)

    # Pick points
    picked, corners = [], []
    def onclick(event):
        x = event.xdata
        y = event.ydata
        if (x is None) or (y is None): return

        # Display the picked point
        picked.append((x, y))
        corners.append((x, y))

        if len(picked) > 1:
            # Draw the line
            temp = np.array(picked)
            ax.plot(temp[:, 0], temp[:, 1])
            ax.figure.canvas.draw_idle()

            # Reset list for future pick events
            del picked[0]

    # Display GUI
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    # Save corner points and image
    if len(corners) > 1: del corners[-1] # Remove last duplicate
    print('Save camera corners ',corners)

def extract_points_3D(velodyne,skip = 10):
    points = np.genfromtxt(velodyne, delimiter=',')[1::skip, :3]
    points = np.asarray(points.tolist())
    print('points ', np.shape(points))

    # Select points within chessboard range
    inrange = np.where((points[:, 0] > 0) &
                       (points[:, 0] < 2.5) &
                       (np.abs(points[:, 1]) < 2.5) &
                       (points[:, 2] < 2))
    points = points[inrange[0]]
    if points.shape[0] > 5:
        print('PCL points available: %d', points.shape)
    else:
        print('Very few PCL points available in range')
        return

    # Color map for the points
    cmap = matplotlib.cm.get_cmap('hsv')
    colors = cmap(points[:, -1] / np.max(points[:, -1]))

    # Setup matplotlib GUI
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Select 3D LiDAR Points', color='white')
    ax.set_axis_off()
    ax.set_facecolor((0, 0, 0))
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],depthshade=False, c=colors, s=2, picker=True)

    # Equalize display aspect ratio for all axes
    max_range = (np.array([points[:, 0].max() - points[:, 0].min(),
                           points[:, 1].max() - points[:, 1].min(),
                           points[:, 2].max() - points[:, 2].min()]).max() / 2.0)
    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    global idx

    def distance(point, event):
        assert point.shape == (3,), "distance: point.shape is wrong: %s, must be (3,)" % point.shape
        # Project 3d data space to 2d data space
        x2, y2, _ = proj3d.proj_transform(point[0], point[1], point[2], plt.gca().get_proj())
        # Convert 2d data space to 2d screen space
        x3, y3 = ax.transData.transform((x2, y2))

        return np.sqrt((x3 - event.x) ** 2 + (y3 - event.y) ** 2)

    def calcClosestDatapoint(X, event):
        distances = [distance(X[i, 0:3], event) for i in range(X.shape[0])]
        return np.argmin(distances)

    def annotatePlot(X, index):
        # If we have previously displayed another label, remove it first
        if hasattr(annotatePlot, 'label'):
            annotatePlot.label.remove()
        # Get data point from array of points X, at position index
        x2, y2, _ = proj3d.proj_transform(X[index, 0], X[index, 1], X[index, 2], ax.get_proj())
        annotatePlot.label = plt.annotate("idx %d" % index,
                                          xy=(x2, y2), xytext=(-20, 20), textcoords='offset points', ha='right',
                                          va='bottom',
                                          bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                                          arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        fig.canvas.draw()

    def onMouseMotion(event):
        """Event that is triggered when mouse is moved. Shows text annotation over data point closest to mouse."""
        closestIndex = calcClosestDatapoint(points, event)
        annotatePlot(points, closestIndex)
        global idx
        idx = closestIndex

    # Pick points
    picked, corners = [], []

    def onpick(event):
        # ind = event.ind[0] #wrong
        global idx
        # print('closestIndex ',idx)
        ind = idx
        x, y, z = event.artist._offsets3d
        # Ignore if same point selected again
        if picked and (x[ind] == picked[-1][0] and y[ind] == picked[-1][1] and z[ind] == picked[-1][2]):
            return

        # Display picked point
        picked.append((x[ind], y[ind], z[ind]))
        corners.append((x[ind], y[ind], z[ind]))
        print('PCL: %s', str(picked[-1]))

        if len(picked) > 1:
            # Draw the line
            temp = np.array(picked)
            ax.plot(temp[:, 0], temp[:, 1], temp[:, 2])
            ax.figure.canvas.draw_idle()
            # Reset list for future pick events
            del picked[0]

    # Display GUI
    fig.canvas.mpl_connect('pick_event', onpick)
    fig.canvas.mpl_connect('motion_notify_event', onMouseMotion)  # on mouse motion
    plt.show()

    # Save corner points
    if len(corners) > 1: del corners[-1]  # Remove last duplicate
    print('Save lidar corners ', corners)

def calibrate():
    '''
    Calibrate the LiDAR and image points using OpenCV PnP RANSAC, Requires minimum 5 point correspondences
    Inputs:
        points2D - [numpy array] - (N, 2) array of image points
        points3D - [numpy array] - (N, 3) array of 3D points
    Outputs:
        Extrinsics saved in PKG_PATH/CALIB_PATH/extrinsics.npz
    '''
    # Load corresponding points
    points2D = np.load('/home/eugen/catkin_ws/src/Camera_Lidar/scripts/img_corners.npy')
    points3D = np.load('/home/eugen/catkin_ws/src/Camera_Lidar/scripts/pcl_corners.npy')
    print('points2D:{},  points3D:{}'.format(np.shape(points2D), np.shape(points3D)))

    if not (points2D.shape[0] >= 5):
        print('PnP RANSAC Requires minimum 5 points')
        return

    # Obtain camera matrix and distortion coefficients
    K = np.array([
        [484.130454, 0.000000, 457.177461],
        [0.000000, 484.452449, 364.861413],
        [0.000000, 0.000000, 1.000000]
    ])
    d = np.array([-0.199619, 0.068964, 0.003371, 0.000296, 0.000000])

    # Estimate extrinsics
    success, rotation_vector, t, _ = cv2.solvePnPRansac(points3D,points2D, K, d,flags=cv2.SOLVEPNP_ITERATIVE)
    print('success ',success)
    # Refine estimate using LM
    if not success:
        print('Initial estimation unsuccessful, skipping refinement')
    elif not hasattr(cv2, 'solvePnPRefineLM'):
        print('solvePnPRefineLM requires OpenCV >= 4.1.1, skipping refinement')
    else:
        print('Here')
        rotation_vector, t = cv2.solvePnPRefineLM(points3D,points2D, K, d,rotation_vector, t)

    # Convert rotation vector
    R = cv2.Rodrigues(rotation_vector)[0]
    euler = None #euler_from_matrix(R)

    # Save extrinsics
    np.savez(os.path.join('/home/eugen/catkin_ws/src/Camera_Lidar/scripts', 'extrinsics.npz'),
             euler=euler, R=R, T=t.T)

    # Display results
    print('Euler angles (RPY):', euler)
    print('Rotation Matrix:', R)
    print('Translation Offsets:', t.T)

def project_point_cloud(velodyne, img):
    '''
    Projects the point cloud on to the image plane using the extrinsics
    Inputs:
        img_msg - [sensor_msgs/Image] - ROS sensor image message
        velodyne - [sensor_msgs/PointCloud2] - ROS velodyne PCL2 message
    Outputs:
        Projected points published on /sensors/camera/camera_lidar topic
    '''

    img = cv2.imread(img)
    # Extract points from message
    skip = 1
    points3D = np.genfromtxt(velodyne, delimiter=',')[1::skip, :3]
    points3D = np.asarray(points3D.tolist())
    print('points3D ', np.shape(points3D))

    # Filter points in front of camera
    #inrange = np.where((points3D[:, 2] > 0) &
    #                   (points3D[:, 2] < 6) &
    #                   (np.abs(points3D[:, 0]) < 6) &
    #                   (np.abs(points3D[:, 1]) < 6))

    inrange = np.where((points3D[:, 0] > 0) &
                       (points3D[:, 0] < 2.5) &
                       (np.abs(points3D[:, 1]) < 2.5) &
                       (points3D[:, 2] < 2))

    max_intensity = np.max(points3D[:, -1])
    points3D = points3D[inrange[0]]
    print('Filtered points3D ', np.shape(points3D))

    # Color map for the points
    cmap = matplotlib.cm.get_cmap('jet')
    colors = cmap(points3D[:, -1] / max_intensity) * 255

    b = np.load('extrinsics.npz')
    print 'b ', np.shape(b)
    euler, R, t = b['euler'],b['R'],b['T']
    print('euler:{}, R:{}, t:{}'.format((euler), np.shape(R), np.shape(t)))

    K = np.array([
        [484.130454, 0.000000, 457.177461],
        [0.000000, 484.452449, 364.861413],
        [0.000000, 0.000000, 1.000000]
    ])
    d = np.array([-0.199619, 0.068964, 0.003371, 0.000296, 0.000000])
    print('R ', np.shape(R))
    transformMatrix = np.hstack((R, t.T))
    print('transformMatrix ',transformMatrix)
    def project3dToPixel(point):
        pointXYZ = np.append(point, [1])
        transformedPoint = transformMatrix.dot(pointXYZ)
        #print('transformedPoint ',transformedPoint)
        #P = K.dot(np.hstack((R, t.T)))
        transformedPoint = np.append(transformedPoint, [1])
        P = np.array([
            [426.423584, 0.000000, 461.366281, 0.000000],
            [0.000000, 432.826263, 369.496601, 0.000000],
            [0.000000, 0.000000, 1.000000, 0.000000]
        ])
        #print('P ',P)
        projected = np.dot(P,transformedPoint)
        return (projected/projected[-1])[:-1]

    # Project to 2D and filter points within image boundaries
    points2D = [project3dToPixel(point) for point in points3D[:, :3]]
    points2D = np.asarray(points2D)
    print 'points2D ', np.shape(points2D)
    print 'points2D[0] ', points2D[0]
    print 'img.shape ',img.shape
    inrange = np.where((points2D[:, 0] >= 0) &
                       (points2D[:, 1] >= 0) &
                       (points2D[:, 0] < img.shape[1]) &
                       (points2D[:, 1] < img.shape[0]))
    points2D = points2D[inrange[0]].round().astype('int')
    print 'points2D final ', np.shape(points2D)

    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, d, (w, h), 1, (w, h))
    # undistort
    '''dst = cv2.undistort(img, K, d, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]'''
    # undistort
    mapx, mapy = cv2.initUndistortRectifyMap(K, d, None, newcameramtx, (w, h), 5)
    dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    #cv2.imshow('calibresult', dst)
    #while True:
    #    k = cv2.waitKey(1)
    #    if k == 27:
    #        cv2.destroyAllWindows()
    #        break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (nCols, nRows) = (7, 5)
    ret, corners = cv2.findChessboardCorners(gray, (nCols, nRows), None)
    print 'ret ',ret
    dimension = 22  # - mm
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, dimension, 0.001)
    if ret == True:
        print("ESC to skip or ENTER to accept")
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(img, (nCols, nRows), corners2, ret)
        cv2.imshow('img', img)

    # Draw the projected 2D points
    for i in range(len(points2D)):
        cv2.circle(img, tuple(points2D[i]), 2, tuple(colors[i]), -1)

    imS = img# cv2.resize(img, (960, 540))
    cv2.imshow('imS', imS)
    while True:
        k = cv2.waitKey(1)
        if k == 27:
            cv2.destroyAllWindows()
            break

def undist():
    img_frame = '/home/eugen/catkin_ws/src/Camera_Lidar/scripts/img_frame.png'
    K = np.array([
        [484.130454, 0.000000, 457.177461],
        [0.000000, 484.452449, 364.861413],
        [0.000000, 0.000000, 1.000000]
    ])
    d = np.array([-0.199619, 0.068964, 0.003371, 0.000296, 0.000000])

    print('K:{}, d:{}'.format(np.shape(K),np.shape(d)))
    mtx, dist = K,d
    # Getting the new optimal camera matrix
    img = cv2.imread(img_frame)
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1,(w, h))


    # Method 1 to undistort the image
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    # Method 2 to undistort the image
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
    dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

    # Undistorting
    #dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # Cropping the image
    x, y, w, h = roi
    dst2 = dst[y:+y + h, x:x + w]
    undistorted = dst
    cv2.imshow('img', img)
    cv2.imshow('undistorted', undistorted)
    cv2.imshow('dst2', dst2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    '''calib = False

    img_frame = '/home/eugen/catkin_ws/src/Camera_Lidar/scripts/img_frame.png'
    pcl_frame = '/home/eugen/catkin_ws/src/Camera_Lidar/scripts/pcl_frame.csv'
    extract_points_3D(pcl_frame)
    if calib:
        #extract_points_2D(img_frame)
        #extract_points_3D(pcl_frame)
        calibrate() #check for R,t here
    else:
        project_point_cloud(pcl_frame,img_frame)'''

    undist()
