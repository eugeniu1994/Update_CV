from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

skip = 15
#source = '/home/eugen/catkin_ws/src/Camera_Lidar/DATA/pcd/0002.csv'
#data = np.genfromtxt(source, delimiter=',')[1::skip,:3]
#print ('data ', np.shape(data))

#x,y,z = data[:,0],data[:,1],data[:,2]

'''fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#ax.scatter(x, y, z, c=c, marker=m)
ax.scatter(x, y, z, s=0.01)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()'''

#---------------------------------------------------------
def fit_plane_scipy(P=None):
    from skspatial.objects import Points, Plane
    from skspatial.plotting import plot_3d

    points = Points([[0, 0, 0], [1, 3, 5], [-5, 6, 3], [3, 6, 7], [-2, 6, 7]]) if P is None else Points(P)

    plane = Plane.best_fit(points)
    plot_3d(
        points.plotter(c='k', s=0.1, depthshade=False),
        plane.plotter(alpha=0.8, lims_x=(-5, 5), lims_y=(-5, 5)),
    )
    plt.show()

#fit_plane_scipy(data)

#---------------------------------------------------------
def fit_plane_1():
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np

    '''N_POINTS = 10
    TARGET_X_SLOPE = 2
    TARGET_y_SLOPE = 3
    TARGET_OFFSET  = 5
    EXTENTS = 5
    NOISE = 5
    
    # create random data
    xs = [np.random.uniform(2*EXTENTS)-EXTENTS for i in range(N_POINTS)]
    ys = [np.random.uniform(2*EXTENTS)-EXTENTS for i in range(N_POINTS)]
    zs = []
    for i in range(N_POINTS):
        zs.append(xs[i]*TARGET_X_SLOPE + ys[i]*TARGET_y_SLOPE + TARGET_OFFSET + np.random.normal(scale=NOISE))'''

    xs,ys,zs = x,y,z

    # plot raw data
    plt.figure()
    ax = plt.subplot(111, projection='3d')
    ax.scatter(xs, ys, zs,s=0.05)

    # do fit
    tmp_A = []
    tmp_b = []
    for i in range(len(xs)):
        tmp_A.append([xs[i], ys[i], 1])
        tmp_b.append(zs[i])
    b = np.matrix(tmp_b).T
    A = np.matrix(tmp_A)

    # Manual solution
    fit = (A.T * A).I * A.T * b
    errors = b - A * fit
    residual = np.linalg.norm(errors)

    # Or use Scipy
    # from scipy.linalg import lstsq
    # fit, residual, rnk, s = lstsq(A, b)

    print("solution:")
    print ("%f x + %f y + %f = z" % (fit[0], fit[1], fit[2]))
    print ("errors:")
    print (errors)
    print ("residual:")
    print (residual)

    # plot plane
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    X,Y = np.meshgrid(np.arange(xlim[0], xlim[1]),
                      np.arange(ylim[0], ylim[1]))
    Z = np.zeros(X.shape)
    for r in range(X.shape[0]):
        for c in range(X.shape[1]):
            Z[r,c] = fit[0] * X[r,c] + fit[1] * Y[r,c] + fit[2]
    ax.plot_wireframe(X,Y,Z, color='k')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
#fit_plane_1()

def test():
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import mpld3
    from mpld3 import plugins

    css = """
    table
    {
      border-collapse: collapse;
    }
    th
    {
      color: #ffffff;
      background-color: #000000;
    }
    td
    {
      background-color: #cccccc;
    }
    table, th, td
    {
      font-family:Arial, Helvetica, sans-serif;
      border: 1px solid black;
      text-align: right;
    }
    """

    fig, ax = plt.subplots()
    ax.grid(True, alpha=0.3)

    N = 50
    df = pd.DataFrame(index=range(N))
    df['x'] = np.random.randn(N)
    df['y'] = np.random.randn(N)
    df['z'] = np.random.randn(N)

    labels = []
    for i in range(N):
        #label = df.ix[[i], :].T
        label = df.iloc[[i], :].T
        label.columns = ['Row {0}'.format(i)]
        # .to_html() is unicode; so make leading 'u' go away with str()
        labels.append(str(label.to_html()))

    points = ax.plot(df.x, df.y, 'o', color='b',
                     mec='k', ms=15, mew=1, alpha=.6)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('HTML tooltips', size=20)

    tooltip = plugins.PointHTMLTooltip(points[0], labels, voffset=10, hoffset=10, css=css)
    plugins.connect(fig, tooltip)

    mpld3.show()
#test()

def pick():
    import matplotlib.pyplot as plt, numpy as np
    from mpl_toolkits.mplot3d import proj3d

    def visualize3DData(X):
        fig = plt.figure(figsize=(16, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], depthshade=False, s=2, picker=True)
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
            annotatePlot.label = plt.annotate("Value %d" % index,
                                              xy=(x2, y2), xytext=(-20, 20), textcoords='offset points', ha='right',
                                              va='bottom',
                                              bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                                              arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            fig.canvas.draw()

        def onMouseMotion(event):
            """Event that is triggered when mouse is moved. Shows text annotation over data point closest to mouse."""
            closestIndex = calcClosestDatapoint(X, event)
            annotatePlot(X, closestIndex)
            global idx
            idx = closestIndex

        # Pick points
        picked, corners = [], []
        def onpick(event):
            #ind = event.ind[0]
            #closestIndex = calcClosestDatapoint(X, event)
            #print('ind ', ind)
            global idx
            #print('closestIndex ',idx)
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

    velodyne = '/home/eugen/catkin_ws/src/Camera_Lidar/scripts/pcl_frame.csv'
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

    X = np.random.random((30, 3))
    X = points
    visualize3DData(X)

#pick()

def test2():
    import numpy as np
    import scipy.optimize

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.gca(projection='3d')

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

    data = np.random.randn(100, 3) / 3
    data[:, 2] /= 10
    c, normal = fitPlaneLTSQ(data)

    # plot fitted plane
    maxx = np.max(data[:, 0])
    maxy = np.max(data[:, 1])
    minx = np.min(data[:, 0])
    miny = np.min(data[:, 1])

    point = np.array([0.0, 0.0, c])
    d = -point.dot(normal)

    # plot original points
    ax.scatter(data[:, 0], data[:, 1], data[:, 2])

    # compute needed points for plane plotting
    xx, yy = np.meshgrid([minx, maxx], [miny, maxy])
    z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]

    # plot plane
    ax.plot_surface(xx, yy, z, alpha=0.2)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

test2()
