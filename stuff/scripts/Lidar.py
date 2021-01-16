import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons
import pcl
from scipy.spatial import distance_matrix
from scipy.optimize import leastsq

def PCA(data, correlation=False, sort=True):
    #data = nx3
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

def distance_matrix_(x1, x2):
    x1 = np.asarray(x1)
    x2 = np.atleast_2d(x2)
    x1_dim = x1.ndim
    x2_dim = x2.ndim
    if x1_dim == 1:
        x1 = x1.reshape(1, 1, x1.shape[0])
    if x1_dim >= 2:
        x1 = x1.reshape(np.prod(x1.shape[:-1]), 1, x1.shape[-1])
    if x2_dim > 2:
        x2 = x2.reshape(np.prod(x2.shape[:-1]), x2.shape[-1])

    diff = x1 - x2
    arr_dist = np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))
    return np.squeeze(arr_dist)

def chessBoard(ax, scale=1., org=[0, 0, 0], R=[0,0,0], plot=True):
    def eulerAnglesToRotationMatrix(theta):
        R_x = np.array([[1, 0, 0],
                        [0, math.cos(theta[0]), -math.sin(theta[0])],
                        [0, math.sin(theta[0]), math.cos(theta[0])]
                        ])

        R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                        [0, 1, 0],
                        [-math.sin(theta[1]), 0, math.cos(theta[1])]
                        ])

        R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                        [math.sin(theta[2]), math.cos(theta[2]), 0],
                        [0, 0, 1]
                        ])

        R = np.dot(R_z, np.dot(R_y, R_x))
        return R

    nCols, nRows, scale, org, R = 8, 11, np.asarray(scale), np.asarray(org), np.asarray(R)
    Rot_matrix = eulerAnglesToRotationMatrix(R)

    X, Y = np.linspace(org[0], org[0] + nCols, num=nCols), np.linspace(org[1], org[1]+nRows, num=nRows)
    #X, Y = np.arange(org[0], org[0]+nCols), np.arange(org[1], org[1]+nRows)

    X, Y = np.meshgrid(X, Y)
    Z = np.full(np.shape(X), org[2])
    colors, colortuple = np.empty(X.shape, dtype=str), ('w', 'k')

    for y in range(nCols):
        for x in range(nRows):
            colors[x, y] = colortuple[(x + y) % len(colortuple)]

    X, Y, Z = X * scale, Y * scale, Z * scale
    corners = np.transpose(np.array([X, Y, Z]), (1, 2, 0))

    init = corners.reshape(-1,3)
    translation = np.mean(init, axis=0)             #get the mean point
    corners = np.subtract(corners,translation)      #substract it from all the other points
    X, Y, Z = np.transpose(np.add(np.dot(corners, Rot_matrix),translation), (2, 0, 1))
    corners = np.transpose(np.array([X, Y, Z]), (1, 2, 0)).reshape(-1,3)
    #print('X:{},Y:{},Z:{}'.format(np.shape(X), np.shape(Y), np.shape(Z)))
    chess, corn = 0,0
    if plot:
        chess = ax.plot_surface(X, Y, Z, facecolors=colors, linewidth=0.2, cmap='gray', alpha=0.75) #ax.plot_wireframe(X, Y, Z, linewidth=.5)
        corn = ax.scatter(corners[:,0], corners[:,1], corners[:,2], c = 'tab:blue', marker='o', s=5 )

    return chess, corn, corners

def getPointCoud(ax, plotInit=False, plotSmoothed=False, plotFinal=True):
    file = '/home/eugen/catkin_ws/src/Camera_Lidar/scripts/pcl_frame.csv'
    file = '/home/eugen/catkin_ws/src/Camera_Lidar/DATA/data.csv'
    file = '/home/eugen/catkin_ws/src/Camera_Lidar/DATA/data_pointCloud/cloud_points_1610553740267486095.npy'
    skip = 1

    points = np.genfromtxt(file, delimiter=',')[1::skip, :3] if file.endswith('.csv') else np.load(file)[1::skip, :]
    print('points ', np.shape(points))
    inrange = np.where((points[:, 0] > 0) &
                       (points[:, 0] < 2) &
                       (np.abs(points[:, 1]) < 2) &
                       (points[:, 2] < 2))
    #points = np.asarray(points[inrange[0]], dtype=np.float32)
    points = np.asarray(points, dtype=np.float32)
    if plotInit:
        p1 = ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.2)
        return [p1], points

    p = pcl.PointCloud(points)

    # smoothing
    fil = p.make_statistical_outlier_filter()
    fil.set_mean_k(50)
    fil.set_std_dev_mul_thresh(1.0)
    smoothed = np.array(fil.filter())
    if plotSmoothed:
        p1 = ax.scatter(smoothed[:, 0], smoothed[:, 1], smoothed[:, 2], s=0.2)
        return [p1],points

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

    def do_ransac_plane_normal_segmentation(point_cloud, input_max_distance, normal_w=0.1):
        segmenter = point_cloud.make_segmenter_normals(ksearch=50)
        segmenter.set_optimize_coefficients(True)
        segmenter.set_model_type(pcl.SACMODEL_NORMAL_PLANE)  # pcl_sac_model_plane

        segmenter.set_normal_distance_weight(normal_w)
        segmenter.set_method_type(pcl.SAC_RANSAC)  # pcl_sac_ransac
        segmenter.set_max_iterations(1000)
        segmenter.set_distance_threshold(input_max_distance)  # 0.03)  #max_distance
        indices, coefficients = segmenter.segment()

        #print('Model coefficients: ' + str(coefficients[0]) + ' ' + str(coefficients[1]) + ' ' + str(coefficients[2]) + ' ' + str(coefficients[3]))
        #print('Model inliers: ' + str(len(indices)))

        inliers = point_cloud.extract(indices, negative=False)
        outliers = point_cloud.extract(indices, negative=True)

        return inliers, outliers

    # RANSAC Plane Segmentation
    points = np.asarray(smoothed, dtype=np.float32)
    p = pcl.PointCloud(points)

    #('w ', array([7.54245755e-03, 3.31826325e-03, 9.82558219e-20]))
    #('v[0] ', array([ 0.99082856, -0.01709094, -0.1340398 ]))

    w = .1
    w = 7.54245755e-03

    #inlier, outliner = do_ransac_plane_segmentation(p, pcl.SACMODEL_PLANE, pcl.SAC_RANSAC, 0.03)
    inlier, outliner = do_ransac_plane_normal_segmentation(point_cloud=p, input_max_distance=0.03, normal_w=w)
    inlier, outliner = np.array(inlier), np.array(outliner)


    #coefficients, inliers, outliers = do_ransac_plane_normal_segmentation(p, 0.01)
    #inlier, outliner = np.array(inliers), np.array(outliers)
    if plotFinal:
        p1 = ax.scatter(outliner[:, 0], outliner[:, 1], outliner[:, 2], c='r', s=0.2)
        p2 = ax.scatter(inlier[:, 0], inlier[:, 1], inlier[:, 2], c='g', s=0.2)

        w, v = PCA(inlier)
        point = np.mean(inlier, axis=0)
        w*=3
        p3=ax.quiver([point[0]], [point[1]], [point[2]], [v[0, :] * np.sqrt(w[0])], [v[1, :] * np.sqrt(w[0])],
                  [v[2, :] * np.sqrt(w[0])], linewidths=(1.8,))

        return [p1,p2,p3], inlier

def FindLidarPoints():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.set_zlabel('Z', fontsize=10)
    plt.subplots_adjust(left=0.05, bottom=0.2)

    marker_size, grid_length = (6, 9), 0.08
    Rx, Ry, Rz = 0, 0, 0
    global chess,corn,t,points,corners3D,axis_on,col
    plotInit, plotSmoothed, plotFinal = False, False, True
    points, Cloud3D=getPointCoud(ax=ax, plotInit=plotInit, plotSmoothed=plotSmoothed, plotFinal=plotFinal)

    axis_on, col = True, False
    R,t=[0,0,0],[0,0,0]
    t = np.mean(Cloud3D, axis=0)/grid_length

    chess, corn, corners3D = chessBoard(ax=ax,org=t,scale=grid_length)

    Rx_Slider = Slider(plt.axes([0.25, 0.15, 0.65, 0.03]), 'Rx', -180, 180.0, valinit=Rx)
    Ry_Slider = Slider(plt.axes([0.25, 0.1, 0.65, 0.03]), 'Ry', -180, 180.0, valinit=Ry)
    Rz_Slider = Slider(plt.axes([0.25, 0.05, 0.65, 0.03]), 'Rz', -180, 180.0, valinit=Rz)

    def update(val):
        Rx,Ry,Rz = Rx_Slider.val, Ry_Slider.val, Rz_Slider.val
        #print('Rx:{}, Ry:{}, Rz:{}'.format(Rx, Ry, Rz))
        global chess,corn,corners3D
        chess.remove()
        corn.remove()
        R = [np.deg2rad(Rx),np.deg2rad(Ry),np.deg2rad(Rz)]
        chess,corn,corners3D = chessBoard(ax=ax, org=t, R=R, scale=grid_length)
        #yellow = ax.scatter(corners3D[:, 0], corners3D[:, 1], corners3D[:, 2], c='yellow', marker='o', s=5)
        fig.canvas.draw_idle()

    Rx_Slider.on_changed(update)
    Ry_Slider.on_changed(update)
    Rz_Slider.on_changed(update)

    check = CheckButtons(plt.axes([0.03, 0.3, 0.15, 0.08]), ('Axes', 'Black'), (True, False))

    def func_CheckButtons(label):
        global col,axis_on
        if label == 'Axes':
            if axis_on:
                ax.set_axis_off()
                axis_on = False
            else:
                ax.set_axis_on()
                axis_on = True
        elif label == 'Black':
            if col:
                col=False
                ax.set_facecolor((1, 1, 1))
            else:
                col=True
                ax.set_facecolor((0, 0, 0))

        fig.canvas.draw_idle()
    check.on_clicked(func_CheckButtons)

    class Index(object):
        step=1
        def Tx(self, event):
            t[0]+=self.step
            update(0)

        def Tx_(self, event):
            t[0] -= self.step
            update(0)

        def Ty(self, event):
            t[1] += self.step
            update(0)

        def Ty_(self, event):
            t[1] -= self.step
            update(0)

        def Tz(self, event):
            t[2] += self.step
            update(0)

        def Tz_(self, event):
            t[2] -= self.step
            update(0)
    callback = Index()

    Tx = Button(plt.axes([0.05, 0.15, 0.04, 0.045]), '+Tx', color='white')
    Tx.on_clicked(callback.Tx)
    Tx_ = Button(plt.axes([0.12, 0.15, 0.04, 0.045]), '-Tx', color='white')
    Tx_.on_clicked(callback.Tx_)

    Ty = Button(plt.axes([0.05, 0.1, 0.04, 0.045]), '+Ty', color='white')
    Ty.on_clicked(callback.Ty)
    Ty_ = Button(plt.axes([0.12, 0.1, 0.04, 0.045]), '-Ty', color='white')
    Ty_.on_clicked(callback.Ty_)

    Tz = Button(plt.axes([0.05, 0.05, 0.04, 0.045]), '+Tz', color='white')
    Tz.on_clicked(callback.Tz)
    Tz_ = Button(plt.axes([0.12, 0.05, 0.04, 0.045]), '-Tz', color='white')
    Tz_.on_clicked(callback.Tz_)

    def getClosestPoints(arg):
        global chess, corn, points, corners3D
        print('Cloud3D:{}, corners3D:{}'.format(np.shape(Cloud3D), np.shape(corners3D)))
        dist_mat, k = distance_matrix(corners3D,Cloud3D), 1
        neighbours = np.argsort(dist_mat, axis=1) [:, 0]
        finaPoints = np.asarray(Cloud3D[neighbours,:]).squeeze()

        points[1].remove()
        chess.remove()
        corn.remove()

        ax.scatter(finaPoints[:, 0], finaPoints[:, 1], finaPoints[:, 2], c='g', marker='o', s=7)
        #ax.plot_wireframe(finaPoints[:, 0], finaPoints[:, 1], finaPoints[:, 2], markersize=2)

        corn = ax.scatter(corners3D[:, 0], corners3D[:, 1], corners3D[:, 2], c='tab:blue', marker='x', s=6)
        fig.canvas.draw_idle()

    savePoints = Button(plt.axes([0.03, 0.45, 0.15, 0.04], ), 'save points', color='white')
    savePoints.on_clicked(getClosestPoints)

    def reset(args):
        ax.cla()
        global chess, corn, corners3D, points
        points, Cloud3D = getPointCoud(ax=ax, plotInit=plotInit, plotSmoothed=plotSmoothed, plotFinal=plotFinal)
        t = np.mean(Cloud3D, axis=0) / grid_length
        chess, corn, corners3D = chessBoard(ax=ax, org=t, scale=grid_length)

        fig.canvas.draw_idle()

    resetBtn = Button(plt.axes([0.03, 0.25, 0.15, 0.04], ), 'reset', color='white')
    resetBtn.on_clicked(reset)

    def auto_fitChessboard(args):
        # estimate 3D-R and 3D-t between chess and PointCloud
        global chess, corn, points, corners3D
        # Inital guess of the transformation
        x0 = np.array([0,0,0,0,0,0])
        def f_min(x):
            global corners3D
            R = [np.deg2rad(x[0]), np.deg2rad(x[1]), np.deg2rad(x[2])]
            t = [x[3],x[4],x[5]]
            _, _, corners3D = chessBoard(ax=ax, org=t, R=R, scale=grid_length, plot=False)

            dist_mat = distance_matrix(corners3D, Cloud3D)
            err_func = dist_mat.sum(axis=1)  # 63 x 1
            #print('x:{},   err_func:{},  sum of errors = {}, dist_mat:{} corners3D:{},Cloud3D:{}'.format([], np.shape(err_func),round(np.sum(err_func),2),np.shape(dist_mat), np.shape(corners3D), np.shape(Cloud3D)))
            return err_func

        sol,status = leastsq(f_min, x0, ftol=1.49012e-06, xtol=1.49012e-06)
        print('sol:{}, status:{}'.format(sol,status))
        #set values of sol to Sliders
        Rx_Slider.set_val(int(sol[0]))
        Ry_Slider.set_val(int(sol[1]))
        Rz_Slider.set_val(int(sol[2]))
        R = [np.deg2rad(sol[0]), np.deg2rad(sol[1]), np.deg2rad(sol[2])]
        t = [sol[3], sol[4], sol[5]]
        chess.remove()
        corn.remove()
        chess, corn, corners3D = chessBoard(ax=ax, org=t, R=R, scale=grid_length)
        fig.canvas.draw_idle()
#37073.64
    fitChessboard = Button(plt.axes([0.03, 0.66, 0.15, 0.04], ), 'auto fit', color='white')
    fitChessboard.on_clicked(auto_fitChessboard)

    radio = RadioButtons(plt.axes([0.03, 0.5, 0.15, 0.15], ), ('Final', 'Smoothed', 'Init'), active=0)

    def colorfunc(label):
        plotInit, plotSmoothed, plotFinal = False, False, False
        if label=='Init':
            plotInit=True
        elif label=='Smoothed':
            plotSmoothed=True
        else:
            plotFinal=True

        global points
        [p.remove() for p in points]
        points,Cloud3D = getPointCoud(ax=ax, plotInit=plotInit, plotSmoothed=plotSmoothed, plotFinal=plotFinal)
        fig.canvas.draw_idle()

    radio.on_clicked(colorfunc)
    plt.show()

if __name__ == '__main__':
    print('Lidar main')

    FindLidarPoints()






