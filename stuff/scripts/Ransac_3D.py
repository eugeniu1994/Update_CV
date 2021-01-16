
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
from mpl_toolkits.mplot3d import Axes3D

random.seed(2021)

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

def distance_function(x1, x2):
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

def euclid_dist(x1, x2):
    return np.sqrt(sum((x1 - x2) ** 2))

def gammaidx(X, k):
    #dist_matrix = np.sqrt(np.sum((X[None, :] - X[:, None])**2, -1))
    dist_matrix = distance_function(X, X)

    near_points = np.argpartition(dist_matrix, k + 1)
    y = []
    for i in range(X.shape[0]):
        y.append(np.sum([euclid_dist(X[i, :], X[j, :]) for j in near_points[i, :k + 1]]))
    y = (1 / k) * np.array(y)

    return y

def Filter(X):
    n = np.shape(X)[0]
    scaler = 5
    gamma_k3 = gammaidx(X, k=3)*scaler
    gamma_k10 = gammaidx(X, k=10)*scaler
    dist_to_mean = np.sqrt(((X - np.mean(X, axis=0)) ** 2).sum(-1))*scaler


    print('gamma_k3:{}, gamma_k10:{}, dist_to_mean:{}'.format(np.shape(gamma_k3), np.shape(gamma_k10), np.shape(dist_to_mean)))
    #ax.scatter(data[:, 0], data[:, 1], data[:, 1], s=gamma_k3, c='g', label='Original data')
    plt.plot(np.linspace(0,len(gamma_k3), len(gamma_k3)),gamma_k3)
    plt.show()
    '''plt.subplot(3, 1, 2)
    plt.scatter(data[:, 0], data[:, 1],data[:, 2], s=gamma_k10, c='g', label='Original data')
    plt.title('Condition b) gamma with k-10')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.scatter(data[:, 0], data[:, 1],data[:, 2], s=dist_to_mean, c='g', label='Original data')
    plt.title('Condition c) dist. to mean score')
    plt.legend()'''

    plt.show()

class RANSAC:
    def __init__(self, point_cloud, max_iterations, distance_ratio_threshold):
        self.point_cloud = point_cloud
        self.max_iterations = max_iterations
        self.distance_ratio_threshold = distance_ratio_threshold

    def run(self, pca = True, equation=False):
        inliers, outliers, (a,b,c,d) = self._ransac_algorithm(self.max_iterations, self.distance_ratio_threshold)

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(inliers.X , inliers.Y,  inliers.Z,  c="blue",alpha=0.3, s=0.05)
        ax.scatter(outliers.X, outliers.Y, outliers.Z, c="red",alpha=0.3, s=0.05)

        ax.set_xlabel('X', fontsize=10)
        ax.set_ylabel('Y', fontsize=10)
        ax.set_zlabel('Z', fontsize=10)

        if pca:
            points = np.array([inliers.X,inliers.Y,inliers.Z]).transpose()
            print('points ', np.shape(points))
            dist_to_mean = np.sqrt(((points - np.mean(points, axis=0)) ** 2).sum(-1))
            mean_dist_to_mean = np.mean(dist_to_mean)
            print('mean_dist_to_mean ',mean_dist_to_mean)
            inrange = np.where(dist_to_mean <= mean_dist_to_mean+.1)
            points = points[inrange[0]]
            print('points ', np.shape(points))
            ax.scatter(points[:,0], points[:,1], points[:,2], c="green", s=1)
            #--------------------------------------------------------------------------

            #inliers, outliers, (a, b, c, d) = self.refit(points)
            #ax.scatter(inliers.X, inliers.Y, inliers.Z, c="tab:blue", s=3)

            w, v = PCA(points)
            print('w:{}, v:{}'.format(np.shape(w),np.shape(v)))
            ax.scatter(v[0,:], v[1,:], v[2,:], s=2)

            #: the normal of the plane is the last eigenvector
            normal = v[:, 2]
            print('normal ',normal)
            #: get a point from the plane
            point = np.mean(points, axis=0)
            print('RANSAC (a:{},b:{},c:{},d:{})'.format(round(a,2),round(b,2),round(c,2),round(d,2)))
            #a, b, c = normal
            #d = -(np.dot(normal, point))
            #print('PCA (a:{},b:{},c:{},d:{})'.format(round(a,2),round(b,2),round(c,2),round(d,2)))
            #ax.quiver([point[0]], [point[1]], [point[2]], [normal[0]], [normal[1]], [normal[2]], linewidths=(1,),edgecolor="tab:blue")
            #ax.quiver([point[0]], [point[1]], [point[2]], [v[0, :]], [v[1, :]], [v[2, :]], linewidths=(.7,))
            w*=3
            ax.quiver([point[0]], [point[1]], [point[2]], [v[0, :]* np.sqrt(w[0])], [v[1, :]* np.sqrt(w[0])], [v[2, :]* np.sqrt(w[0])], linewidths=(1.8,))

            #self.fitPlane(ax=ax,data=points)
            self.fitConvexHull(points=points,ax=ax)

            '''if equation:
                return a, b, c, d
            else:
                return point, normal'''

        #plt.show()

        #fig = plt.figure()
        #plt.plot(np.linspace(0, len(dist_to_mean), len(dist_to_mean)), dist_to_mean, label='dist to mean')
        #plt.legend()
        plt.show()

    def _visualize_point_cloud(self):
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(self.point_cloud.X , self.point_cloud.Y, self.point_cloud.Z, s=0.2)
        ax.set_xlabel('X', fontsize=10)
        ax.set_ylabel('Y', fontsize=10)
        ax.set_zlabel('Z', fontsize=10)

        plt.show()

    def refit(self,points, iters = 10):
        inliers_result = set()

        marker_size = (6, 8)
        marker_w = 0.075 * marker_size[1]
        marker_h = 0.075 * marker_size[0]
        area = marker_w * marker_h
        print('marker_w:{}, marker_h:{}, area:{}'.format(marker_w, marker_h, area))

        while max_iterations:
            max_iterations -= 1
            # Add 3 random indexes
            # random.seed()
            inliers = []
            while len(inliers) < 3:
                random_index = random.randint(0, len(self.point_cloud.X) - 1)
                inliers.append(random_index)
            # print(inliers)
            try:
                # In case of *.xyz data
                x1, y1, z1, _, _, _ = point_cloud.loc[inliers[0]]
                x2, y2, z2, _, _, _ = point_cloud.loc[inliers[1]]
                x3, y3, z3, _, _, _ = point_cloud.loc[inliers[2]]
            except:
                # In case of *.pcd data
                x1, y1, z1 = point_cloud.loc[inliers[0]]
                x2, y2, z2 = point_cloud.loc[inliers[1]]
                x3, y3, z3 = point_cloud.loc[inliers[2]]
            # Plane Equation --> ax + by + cz + d = 0
            # Value of Constants for inlier plane
            a = (y2 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1)
            b = (z2 - z1) * (x3 - x1) - (x2 - x1) * (z3 - z1)
            c = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
            d = -(a * x1 + b * y1 + c * z1)
            plane_lenght = max(0.1, math.sqrt(a * a + b * b + c * c))
            # plane_lenght = math.sqrt(a*a + b*b + c*c)

            if 1:
                for point in self.point_cloud.iterrows():
                    index = point[0]
                    # Skip iteration if point matches the randomly generated inlier point
                    if index in inliers:
                        continue
                    try:
                        # In case of *.xyz data
                        x, y, z, _, _, _ = point[1]
                    except:
                        # In case of *.pcd data
                        x, y, z = point[1]

                    # Calculate the distance of the point to the inlier plane
                    distance = math.fabs(a * x + b * y + c * z + d) / plane_lenght
                    # Add the point as inlier, if within the threshold distancec ratio
                    # final = math.sqrt(a * a + b * b + c * c)
                    # print(np.fabs(final-1.82))
                    if distance <= distance_ratio_threshold:  # and np.abs(final-1.82)<1:
                        inliers.append(index)
                # Update the set for retaining the maximum number of inlier points
                if len(inliers) > len(inliers_result):
                    # inliers_result.clear()
                    # inliers_result = set()
                    inliers_result = inliers

        print('a:{}, b:{}, c:{}, d:{}'.format(round(a, 2), round(b, 2), round(c, 2), round(d, 2)))
        print('final dist ', math.sqrt(a * a + b * b + c * c))
        # 'final dist ', 1.827599381385787

        # Segregate inliers and outliers from the point cloud
        inlier_points = pd.DataFrame(columns={"X", "Y", "Z"})
        outlier_points = pd.DataFrame(columns={"X", "Y", "Z"})
        for point in point_cloud.iterrows():
            if point[0] in inliers_result:
                inlier_points = inlier_points.append({"X": point[1]["X"],
                                                      "Y": point[1]["Y"],
                                                      "Z": point[1]["Z"]}, ignore_index=True)
                continue
            outlier_points = outlier_points.append({"X": point[1]["X"],
                                                    "Y": point[1]["Y"],
                                                    "Z": point[1]["Z"]}, ignore_index=True)
        return inlier_points, outlier_points, (a, b, c, d)

    def _ransac_algorithm(self, max_iterations, distance_ratio_threshold):
        inliers_result = set()

        marker_size = (6,8)
        marker_w = 0.075 * marker_size[1]
        marker_h = 0.075 * marker_size[0]
        area = marker_w * marker_h
        print('marker_w:{}, marker_h:{}, area:{}'.format(marker_w,marker_h,area))

        while max_iterations:
            max_iterations -= 1
            # Add 3 random indexes
            #random.seed()
            inliers = []
            while len(inliers) < 3:
                random_index = random.randint(0, len(self.point_cloud.X)-1)
                inliers.append(random_index)
            # print(inliers)
            try:
                # In case of *.xyz data
                x1, y1, z1, _, _, _ = point_cloud.loc[inliers[0]]
                x2, y2, z2, _, _, _ = point_cloud.loc[inliers[1]]
                x3, y3, z3, _, _, _ = point_cloud.loc[inliers[2]]
            except:
                # In case of *.pcd data
                x1, y1, z1 = point_cloud.loc[inliers[0]]
                x2, y2, z2 = point_cloud.loc[inliers[1]]
                x3, y3, z3 = point_cloud.loc[inliers[2]]
            # Plane Equation --> ax + by + cz + d = 0
            # Value of Constants for inlier plane
            a = (y2 - y1)*(z3 - z1) - (z2 - z1)*(y3 - y1)
            b = (z2 - z1)*(x3 - x1) - (x2 - x1)*(z3 - z1)
            c = (x2 - x1)*(y3 - y1) - (y2 - y1)*(x3 - x1)
            d = -(a*x1 + b*y1 + c*z1)
            #plane_lenght = max(0.1, math.sqrt(a*a + b*b + c*c))
            plane_lenght = max(0.01, math.sqrt(a*a + b*b + c*c))

            if 1:
                for point in self.point_cloud.iterrows():
                    index = point[0]
                    # Skip iteration if point matches the randomly generated inlier point
                    if index in inliers:
                        continue
                    try:
                        # In case of *.xyz data
                        x, y, z, _, _, _ = point[1]
                    except:
                        # In case of *.pcd data
                        x, y, z = point[1]

                    # Calculate the distance of the point to the inlier plane
                    distance = math.fabs(a*x + b*y + c*z + d)/plane_lenght
                    # Add the point as inlier, if within the threshold distancec ratio
                    if distance <= distance_ratio_threshold:
                        inliers.append(index)
                # Update the set for retaining the maximum number of inlier points
                if len(inliers) > len(inliers_result):
                    inliers_result = inliers

        print('a:{}, b:{}, c:{}, d:{}'.format(round(a,2),round(b,2),round(c,2),round(d,2)))
        print('final dist ',math.sqrt(a*a + b*b + c*c))
        #'final dist ', 1.827599381385787

        # Segregate inliers and outliers from the point cloud
        inlier_points = pd.DataFrame(columns={"X", "Y", "Z"})
        outlier_points = pd.DataFrame(columns={"X", "Y", "Z"})
        for point in point_cloud.iterrows():
            if point[0] in inliers_result:
                inlier_points = inlier_points.append({"X": point[1]["X"],
                                                      "Y": point[1]["Y"],
                                                      "Z": point[1]["Z"]}, ignore_index=True)
                continue
            outlier_points = outlier_points.append({"X": point[1]["X"],
                                                    "Y": point[1]["Y"],
                                                    "Z": point[1]["Z"]}, ignore_index=True)
        #refit again here, using only inlier points
        #compute boundary -> check if correspond to real board boundaries
        return inlier_points, outlier_points, (a,b,c,d)

    def fitPlane(self,ax, data, stride=2):
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

        ax.plot_wireframe(xx, yy, z, alpha=0.9,rstride=stride, cstride=stride)

    def fitConvexHull(self,points, ax):
        from scipy.spatial import ConvexHull
        from scipy.spatial import Delaunay

        mean_z = np.mean(points[:,-1])
        hull = ConvexHull(points)

        for simplex in hull.simplices:
            ax.plot(points[simplex, 0], points[simplex, 1], points[simplex, 2], 'k-',lw=.5)

        #ax.plot(points[hull.vertices, 0], points[hull.vertices, 1],points[hull.vertices, 2], 'r--', lw=2)

def test(): #LS
    import numpy as np
    # coordinates (XYZ) of C1, C2, C4 and C5
    XYZ = np.array([
        [0.274791784, -1.001679346, -1.851320839, 0.365840754],
        [-1.155674199, -1.215133985, 0.053119249, 1.162878076],
        [1.216239624, 0.764265677, 0.956099579, 1.198231236]])

    # Inital guess of the plane
    p0 = [0.506645455682, -0.185724560275, -1.43998120646, 1.37626378129]

    def f_min(X, p):
        plane_xyz = p[0:3]
        distance = (plane_xyz * X.T).sum(axis=1) + p[3]
        return distance / np.linalg.norm(plane_xyz)

    def residuals(params, signal, X):
        return f_min(X, params)

    from scipy.optimize import leastsq
    sol = leastsq(residuals, p0, args=(None, XYZ))[0]

    print("Solution: ", sol)
    print("Old Error: ", (f_min(XYZ, p0) ** 2).sum())
    print("New Error: ", (f_min(XYZ, sol) ** 2).sum())

    def fit_plane(voxels, iterations=50, inlier_thresh=10):  # voxels : x,y,z
        inliers, planes = [], []
        xy1 = np.concatenate([voxels[:, :-1], np.ones((voxels.shape[0], 1))], axis=1)
        z = voxels[:, -1].reshape(-1, 1)
        for _ in range(iterations):
            random_pts = voxels[np.random.choice(voxels.shape[0], voxels.shape[1] * 10, replace=False), :]
            plane_transformation, residual = fit_pts_to_plane(random_pts)
            inliers.append(((z - np.matmul(xy1, plane_transformation)) <= inlier_thresh).sum())
            planes.append(plane_transformation)
        return planes[np.array(inliers).argmax()], inliers, planes

    def fit_pts_to_plane(voxels):  # x y z  (m x 3)
        # https: // math.stackexchange.com / questions / 99299 / best - fitting - plane - given - a - set - of - points
        xy1 = np.concatenate([voxels[:, :-1], np.ones((voxels.shape[0], 1))], axis=1)
        z = voxels[:, -1].reshape(-1, 1)
        fit = np.matmul(np.matmul(np.linalg.inv(np.matmul(xy1.T, xy1)), xy1.T), z)
        errors = z - np.matmul(xy1, fit)
        residual = np.linalg.norm(errors)
        return fit, residual

def ConvexHull_(points=None):
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from scipy.spatial import ConvexHull

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    points = np.array([[0, 0, 0],
                       [4, 0, 0],
                       [4, 4, 0],
                       [0, 4, 0],
                       [0, 0, 4],
                       [4, 0, 4],
                       [4, 4, 4],
                       [0, 4, 4]])

    hull = ConvexHull(points)

    #edges = zip(*points)
    edges = list(zip(*points))

    for i in hull.simplices:
        plt.plot(points[i, 0], points[i, 1], points[i, 2], 'r-')

    ax.plot(edges[0], edges[1], edges[2], 'bo')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.set_xlim3d(-5, 5)
    ax.set_ylim3d(-5, 5)
    ax.set_zlim3d(-5, 5)

    plt.show()

if __name__ == "__main__":
    file = '/src/Camera_Lidar/DATA/point_cloud_data_sample.xyz'
    file = '/home/eugen/catkin_ws/src/Camera_Lidar/scripts/pcl_frame.csv'

    skip = 1
    data = np.genfromtxt(file, delimiter=',')[1::skip, :3]
    #point_cloud = pd.read_csv(file, delimiter=" ", nrows=5000)
    points = data
    inrange = np.where((points[:, 0] > 0) &
                       (points[:, 0] < 2) &
                       (np.abs(points[:, 1]) < 2) &
                       (points[:, 2] < 2))
    points = points[inrange[0]]
    point_cloud = pd.DataFrame(points, columns={"X" ,"Y" ,"Z"})
    print('point_cloud ', np.shape(point_cloud))

    #APPLICATION = RANSAC(point_cloud, max_iterations=50, distance_ratio_threshold=.001) #works
    #APPLICATION = RANSAC(point_cloud, max_iterations=50, distance_ratio_threshold=.02)
    #APPLICATION._visualize_point_cloud()
    #APPLICATION.run(pca=True)
    #test()

    ConvexHull_()
