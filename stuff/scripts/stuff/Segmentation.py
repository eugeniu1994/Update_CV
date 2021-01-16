#import pcl
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def chessBoard(ax, scale=1., org=[0, 0, 0]):
    nCols, nRows, scale, org = 6 + 1, 8 + 1, np.asarray(scale), np.asarray(org)
    print('origin ', org[0])
    X, Y = np.arange(org[0], nCols), np.arange(org[1], nRows)
    X, Y = np.meshgrid(X, Y)
    Z = np.full(np.shape(X), org[2])
    colors, colortuple = np.empty(X.shape, dtype=str), ('w', 'k')
    for y in range(nCols):
        for x in range(nRows):
            colors[x, y] = colortuple[(x + y) % len(colortuple)]

    print('x:{},y:{},z:{}'.format(np.shape(X), np.shape(Y), np.shape(Z)))
    X, Y, Z = X * scale, Y * scale, Z * scale
    surf = ax.plot_surface(X, Y, Z, facecolors=colors, linewidth=0, cmap='gray', alpha=0.6)

def Test():
    import pcl
    import numpy as np

    file = '/home/eugen/catkin_ws/src/Camera_Lidar/scripts/pcl_frame.csv'
    skip = 1
    points = np.genfromtxt(file, delimiter=',')[1::skip, :3]
    inrange = np.where((points[:, 0] > 0) &
                       (points[:, 0] < 2) &
                       (np.abs(points[:, 1]) < 2) &
                       (points[:, 2] < 2))
    points = np.asarray(points[inrange[0]], dtype=np.float32)

    '''fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(points[:,0], points[:,1], points[:,2], s=0.2)
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.set_zlabel('Z', fontsize=10)
    #plt.show()'''

    p = pcl.PointCloud(points)

    # smoothing
    fil = p.make_statistical_outlier_filter()
    fil.set_mean_k(50)
    fil.set_std_dev_mul_thresh(1.0)
    smoothed = np.array(fil.filter())
    print('smoothed ', np.shape(smoothed))
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(smoothed[:, 0], smoothed[:, 1], smoothed[:, 2], s=0.2)
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.set_zlabel('Z', fontsize=10)
    #plt.show()

    # find planes
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

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(outliner[:, 0], outliner[:, 1], outliner[:, 2], c='r', s=0.2)
    ax.scatter(inlier[:, 0], inlier[:, 1], inlier[:, 2], c='g', s=0.2)
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.set_zlabel('Z', fontsize=10)

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

    coefficients, inliers, outliers = do_ransac_plane_normal_segmentation(p, 0.05)
    inlier, outliner = np.array(inliers), np.array(outliers)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(outliner[:, 0], outliner[:, 1], outliner[:, 2], c='r', s=0.2)
    ax.scatter(inlier[:, 0], inlier[:, 1], inlier[:, 2], c='g', s=0.2)
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.set_zlabel('Z', fontsize=10)

    a, b, c, d = coefficients[0], coefficients[1], coefficients[2], coefficients[3]
    data = inlier

    def fitPlane(ax, data):
        def fitPlaneLTSQ(XYZ):
            (rows, cols) = XYZ.shape
            G = np.ones((rows, 3))
            G[:, 0] = XYZ[:, 0]  # X
            G[:, 1] = XYZ[:, 1]  # Y
            Z = XYZ[:, 2]
            (a, b, c), resid, rank, s = np.linalg.lstsq(G, Z, rcond=-1)
            normal = (a, b, -1)
            nn = np.linalg.norm(normal)
            normal = normal / nn
            return (c, normal)

        c, normal = fitPlaneLTSQ(data)

        # plot fitted plane
        maxx = np.max(data[:, 0])
        maxy = np.max(data[:, 1])
        minx = np.min(data[:, 0])
        miny = np.min(data[:, 1])

        point = np.array([0.0, 0.0, c])
        d = -point.dot(normal)

        # compute needed points for plane plotting
        xx, yy = np.meshgrid([minx, maxx], [miny, maxy])
        z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]

        ax.plot_wireframe(xx, yy, z, alpha=0.9)

    # fitPlane(ax,data=data)

    def fitConvexHull(points, ax):
        from scipy.spatial import ConvexHull
        from scipy.spatial import Delaunay
        hull = ConvexHull(points)
        for simplex in hull.simplices:
            ax.plot(points[simplex, 0], points[simplex, 1], points[simplex, 2], 'k-', lw=.5)

        # ax.plot(points[hull.vertices, 0], points[hull.vertices, 1],points[hull.vertices, 2], 'r--', lw=2)

    # fitConvexHull(data,ax)

    marker_size, grid_length = (6, 8), 0.075
    w, h = grid_length * marker_size[1], grid_length * marker_size[0]

    # user adjust pose
    # automatically fit it into the pointcloud using some homographies
    chessBoard(ax, scale=grid_length)

    plt.show()

def testInteractive_():
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.widgets import Slider, Button, RadioButtons

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plt.subplots_adjust(left=0.25, bottom=0.25)
    t = np.arange(0.0, 1.0, 0.001)
    ### create constant z-coordinate
    z = np.zeros_like(t)  # <------------ here
    a0 = 5
    f0 = 3
    s = a0 * np.sin(2 * np.pi * f0 * t)
    l, = plt.plot(t, s, lw=2, color='red')
    l2, = plt.plot(t, s, lw=2, color='tab:blue')
    plt.axis([0, 1, -10, 10])

    axfreq = plt.axes([0.25, 0.1, 0.65, 0.03])
    axamp = plt.axes([0.25, 0.15, 0.65, 0.03])

    sfreq = Slider(axfreq, 'Freq', 0.1, 30.0, valinit=f0)
    samp = Slider(axamp, 'Amp', 0.1, 10.0, valinit=a0)

    def update(val):
        amp = samp.val
        freq = sfreq.val
        # set constant z coordinate
        l.set_data(t, z)  # <------------ here
        # set values to y-coordinate
        l.set_3d_properties(amp * np.sin(2 * np.pi * freq * t), zdir="y")  # <------------ here
        fig.canvas.draw_idle()

    sfreq.on_changed(update)
    samp.on_changed(update)

    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', hovercolor='0.975')

    def reset(event):
        sfreq.reset()
        samp.reset()

    button.on_clicked(reset)

    rax = plt.axes([0.025, 0.5, 0.15, 0.15],)
    radio = RadioButtons(rax, ('red', 'blue', 'green'), active=0)

    def colorfunc(label):
        l.set_color(label)
        fig.canvas.draw_idle()

    radio.on_clicked(colorfunc)

    plt.show()

def testInteractive():
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.widgets import Slider, Button, RadioButtons

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plt.subplots_adjust(left=0.25, bottom=0.25)
    t = np.arange(0.0, 1.0, 0.001)
    ### create constant z-coordinate
    z = np.zeros_like(t)  # <------------ here
    a0 = 5
    f0 = 3
    s = a0 * np.sin(2 * np.pi * f0 * t)
    l, = plt.plot(t, s, lw=2, color='red')
    l2, = plt.plot(t, s, lw=2, color='tab:blue')
    plt.axis([0, 1, -10, 10])

    axfreq = plt.axes([0.25, 0.1, 0.65, 0.03])
    axamp = plt.axes([0.25, 0.15, 0.65, 0.03])

    sfreq = Slider(axfreq, 'Freq', 0.1, 30.0, valinit=f0)
    samp = Slider(axamp, 'Amp', 0.1, 10.0, valinit=a0)

    def update(val):
        amp = samp.val
        freq = sfreq.val
        # set constant z coordinate
        l.set_data(t, z)  # <------------ here
        # set values to y-coordinate
        l.set_3d_properties(amp * np.sin(2 * np.pi * freq * t), zdir="y")  # <------------ here
        fig.canvas.draw_idle()

    sfreq.on_changed(update)
    samp.on_changed(update)

    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', hovercolor='0.975')

    def reset(event):
        sfreq.reset()
        samp.reset()

    button.on_clicked(reset)

    rax = plt.axes([0.025, 0.5, 0.15, 0.15],)
    radio = RadioButtons(rax, ('red', 'blue', 'green'), active=0)

    def colorfunc(label):
        l.set_color(label)
        fig.canvas.draw_idle()

    radio.on_clicked(colorfunc)

    plt.show()

def Test_ICP():
    import math
    from sklearn.neighbors import NearestNeighbors

    def euclidean_distance(point1, point2):
        a = np.array(point1)
        b = np.array(point2)
        return np.linalg.norm(a - b, ord=2)

    def point_based_matching(point_pairs):
        """:param point_pairs: the matched point pairs [((x1, y1), (x1', y1')), ..., ((xi, yi), (xi', yi')), ...]
        :return: the rotation angle and the 2D translation (x, y) to be applied for matching the given pairs of points"""

        x_mean,y_mean,xp_mean,yp_mean = 0,0,0,0
        n = len(point_pairs)

        if n == 0:
            return None, None, None

        for pair in point_pairs:
            (x, y), (xp, yp) = pair

            x_mean += x
            y_mean += y
            xp_mean += xp
            yp_mean += yp

        x_mean /= n
        y_mean /= n
        xp_mean /= n
        yp_mean /= n

        s_x_xp,s_y_yp, s_x_yp,s_y_xp = 0,0 ,0,0
        for pair in point_pairs:
            (x, y), (xp, yp) = pair

            s_x_xp += (x - x_mean) * (xp - xp_mean)
            s_y_yp += (y - y_mean) * (yp - yp_mean)
            s_x_yp += (x - x_mean) * (yp - yp_mean)
            s_y_xp += (y - y_mean) * (xp - xp_mean)

        rot_angle = math.atan2(s_x_yp - s_y_xp, s_x_xp + s_y_yp)
        t_x = xp_mean - (x_mean * math.cos(rot_angle) - y_mean * math.sin(rot_angle))
        t_y = yp_mean - (x_mean * math.sin(rot_angle) + y_mean * math.cos(rot_angle))

        return rot_angle, t_x, t_y

    def icp(reference_points, points, max_iterations=100, distance_threshold=0.2,
            convergence_translation_threshold=1e-3,
            convergence_rotation_threshold=1e-4, point_pairs_threshold=10, verbose=False):
        """
        :param reference_points: the reference point set as a numpy array (N x 2)
        :param points: the point that should be aligned to the reference_points set as a numpy array (M x 2)
        :param distance_threshold: the distance threshold between two points in order to be considered as a pair
        :param convergence_translation_threshold: the threshold for the translation parameters (x and y) for the
                                                  transformation to be considered converged
        :param convergence_rotation_threshold: the threshold for the rotation angle (in rad) for the transformation
                                                   to be considered converged
        :param point_pairs_threshold: the minimum number of point pairs the should exist
        :param verbose: whether to print informative messages about the process (default: False)
        :return: the transformation history as a list of numpy arrays containing the rotation (R) and translation (T)
                 transformation in each iteration in the format [R | T] and the aligned points as a numpy array M x 2
        """

        transformation_history = []

        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(reference_points)

        for iter_num in range(max_iterations):
            if verbose:
                print('------ iteration', iter_num, '------')

            closest_point_pairs = []  # list of point correspondences for closest point rule

            distances, indices = nbrs.kneighbors(points)
            for nn_index in range(len(distances)):
                if distances[nn_index][0] < distance_threshold:
                    closest_point_pairs.append((points[nn_index], reference_points[indices[nn_index][0]]))

            # if only few point pairs, stop process
            if verbose:
                print('number of pairs found:', len(closest_point_pairs))
            if len(closest_point_pairs) < point_pairs_threshold:
                if verbose:
                    print('No better solution can be found (very few point pairs)!')
                break

            # compute translation and rotation using point correspondences
            closest_rot_angle, closest_translation_x, closest_translation_y = point_based_matching(closest_point_pairs)
            if closest_rot_angle is not None:
                if verbose:
                    print('Rotation:', math.degrees(closest_rot_angle), 'degrees')
                    print('Translation:', closest_translation_x, closest_translation_y)
            if closest_rot_angle is None or closest_translation_x is None or closest_translation_y is None:
                if verbose:
                    print('No better solution can be found!')
                break

            # transform 'points' (using the calculated rotation and translation)
            c, s = math.cos(closest_rot_angle), math.sin(closest_rot_angle)
            rot = np.array([[c, -s],
                            [s, c]])
            aligned_points = np.dot(points, rot.T)
            aligned_points[:, 0] += closest_translation_x
            aligned_points[:, 1] += closest_translation_y

            # update 'points' for the next iteration
            points = aligned_points

            # update transformation history
            transformation_history.append(
                np.hstack((rot, np.array([[closest_translation_x], [closest_translation_y]]))))

            # check convergence
            if (abs(closest_rot_angle) < convergence_rotation_threshold) \
                    and (abs(closest_translation_x) < convergence_translation_threshold) \
                    and (abs(closest_translation_y) < convergence_translation_threshold):
                if verbose:
                    print('Converged!')
                break

        return transformation_history, points

    np.random.seed(12345)

    # create a set of points to be the reference for ICP
    xs = np.random.random_sample((50, 1))
    ys = np.random.random_sample((50, 1))
    reference_points = np.hstack((xs, ys))

    # 1. remove some points
    points_to_be_aligned = reference_points[1:47]

    # 2. apply rotation to the new point set
    theta = math.radians(12)
    c, s = math.cos(theta), math.sin(theta)
    rot = np.array([[c, -s],
                    [s, c]])
    points_to_be_aligned = np.dot(points_to_be_aligned, rot)

    # 3. apply translation to the new point set
    points_to_be_aligned += np.array([np.random.random_sample(), np.random.random_sample()])

    # run icp
    transformation_history, aligned_points = icp(reference_points, points_to_be_aligned, verbose=True)

    # show results
    plt.plot(reference_points[:, 0], reference_points[:, 1], 'rx', label='reference points')
    plt.plot(points_to_be_aligned[:, 0], points_to_be_aligned[:, 1], 'b1', label='points to be aligned')
    plt.plot(aligned_points[:, 0], aligned_points[:, 1], 'g+', label='aligned points')
    plt.legend()
    plt.show()

if __name__ == "__main__":

    #Test()
    Test_ICP()

    """from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import numpy as np

    n_radii = 8
    n_angles = 36

    # Make radii and angles spaces (radius r=0 omitted to eliminate duplication).
    radii = np.linspace(0.125, 1.0, n_radii)
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

    # Repeat all angles for each radius.
    angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)

    # Convert polar (radii, angles) coords to cartesian (x, y) coords.
    # (0, 0) is manually added at this stage,  so there will be no duplicate
    # points in the (x, y) plane.
    x = np.append(0, (radii * np.cos(angles)).flatten())
    y = np.append(0, (radii * np.sin(angles)).flatten())

    # Compute z to make the pringle surface.
    z = np.sin(-x * y)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    #ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)
    chessBoard(ax)

    steps = 100
    theta = np.linspace(0, 2 * np.pi, steps)
    r_max = 1.2
    x = np.zeros_like(theta)
    y = r_max * np.cos(theta)
    z = r_max * np.sin(theta)
    ax.plot(x, y, z, 'r')
    ax.plot(y, x, z, 'g')
    ax.plot(z, y, x, 'b')
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.set_zlabel('Z', fontsize=10)

    scale = 1.08
    ax.quiver((0,), (0), (0),
              (0), (0), (r_max), color=('c'))
    ax.text(0, 0, r_max * scale, 'Z Theta', weight='bold')

    ax.quiver((0), (0), (0),
              (0), (r_max), (0), color=('m'))
    ax.text(0, r_max * scale, 0, 'Y', weight='bold')

    ax.quiver((0), (0), (0),
              (r_max), (0), (0), color=('y'))
    ax.text(r_max * scale, 0, 0, 'X', weight='bold')

    plt.show()


    #testInteractive()

    from matplotlib.widgets import Button,Slider
    import mpl_interactions.ipyplot as iplt

    freqs = np.arange(2, 20, 3)

    #fig, ax = plt.subplots()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.25)

    TWOPI = 2 * np.pi
    t = np.arange(0.0, TWOPI, 0.001)
    t2 = np.arange(-TWOPI, 0, 0.001)
    initial_amp = .5
    s = initial_amp/100 * np.sin(t)
    #l, = ax.plot(t, t2, s, lw=2)
    #l2, = ax.plot(t, t2, s, lw=2)

    l, = plt.plot(t, t2,s, lw=2)
    l2, = plt.plot(t2, t,s, lw=2)
    #l, = plt.plot(t, s, lw=2)
    #Ry_,Rx_ = 0,0
    x = np.linspace(0, 2 * np.pi, 200)

    def f(x, Rx,Ry):
        print('Rx:{}, Ry:{}'.format(Rx,Ry))
        Rx *= .01
        Ry *= .01
        d2 = np.cos(x*Ry)
        d1 = np.sin(x * Rx)
        rv = np.hstack((d1,d2)).reshape(-1,2)
        print('rv ',np.shape(rv))
        return d1

    class Index(object):
        ind = 0

        def Tx(self, event):
            #self.ind += 1
            #i = self.ind % len(freqs)
            #ydata = np.sin(2 * np.pi * freqs[i] * t)
            #l.set_ydata(ydata)
            #plt.draw()
            print('+Tx')

        def Tx_(self, event):
            #self.ind -= 1
            #i = self.ind % len(freqs)
            #ydata = np.sin(2 * np.pi * freqs[i] * t)
            #l.set_ydata(ydata)
            #plt.draw()
            print('-Tx')

        def Ty(self, event):
            print('Ty')

        def Ty_(self, event):
            print('Ty_')

        def Tz(self, event):
            print('Tz')

        def Tz_(self, event):
            print('Tz_')

    callback = Index()

    sliderX = Slider(plt.axes([0.35, 0.1, 0.55, 0.03]), label='Rx', valmin=0.0, valmax=360, valinit=0)
    def updateX(val):
        print('received val :{}, sliderX.val:{}'.format(val,sliderX.val))
        # amp is the current value of the slider
        amp = sliderX.val/100
        #print('sliderX ',amp)

        l2.set_xdata(amp * np.sin(t))

        # redraw canvas while idle
        fig.canvas.draw_idle()
        plt.draw()
        plt.pause(0.00001)

        plt.gcf().canvas.flush_events()
        plt.show(block=False)

        fig.canvas.draw()
        fig.canvas.flush_events()


    # call update function on slider value change
    sliderX.on_changed(updateX)
    #controls = iplt.plot(x, f, Rx=sliderX, ax=ax)

    '''def f2(x, Rx=Rx_,Ry=Ry_):
        print('Ry:{}'.format(Ry))
        Ry*=.1
        return np.sin(x * Ry)'''

    sliderY = Slider(plt.axes([0.35, 0.055, 0.55, 0.03]), label="Ry", valmin=0.0, valmax=360, valinit=0)
    def updateY(val):
        # amp is the current value of the slider
        amp = sliderY.val/100
        print('sliderY ',amp)
        # update curve
        l.set_ydata(amp * np.sin(t))
        # redraw canvas while idle
        fig.canvas.draw_idle()
    # call update function on slider value change
    #sliderY.on_changed(updateY)

    #controls = iplt.plot(x, f, Rx=sliderX, Ry=sliderY, ax=ax)
    #controls = iplt.

    #controls2 = iplt.plot(x, f2, Ry=sliderY, ax=ax)
    plt.show()
    sliderZ = Slider(plt.axes([0.35, 0.01, 0.55, 0.03]), label="Rz", valmin=0.0, valmax=360, valinit=0)
    def updateZ(val):
        # amp is the current value of the slider
        amp = sliderZ.val / 100
        print('sliderZ ', amp)
        # update curve
        l.set_zdata(amp * np.sin(t))
        # redraw canvas while idle
        fig.canvas.draw_idle()
    # call update function on slider value change
    sliderZ.on_changed(updateZ)


    Tx = Button(plt.axes([0.05, 0.1, 0.04, 0.045]), '+Tx')
    Tx.on_clicked(callback.Tx)
    Tx_ = Button(plt.axes([0.12, 0.1, 0.04, 0.045]), '-Tx')
    Tx_.on_clicked(callback.Tx_)

    Ty = Button(plt.axes([0.05, 0.055, 0.04, 0.045]), '+Ty')
    Ty.on_clicked(callback.Ty)
    Ty_ = Button(plt.axes([0.12, 0.055, 0.04, 0.045]), '-Ty')
    Ty_.on_clicked(callback.Ty_)

    Tz = Button(plt.axes([0.05, 0.01, 0.04, 0.045]), '+Tz')
    Tz.on_clicked(callback.Tz)
    Tz_ = Button(plt.axes([0.12, 0.01, 0.04, 0.045]), '-Tz')
    Tz_.on_clicked(callback.Tz_)

    plt.show()



    '''"""
