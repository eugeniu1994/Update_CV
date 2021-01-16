import numpy as np
from src.Camera_Lidar.scripts.Lidar.Utils import get_default_params
import os
import shutil

params = get_default_params()
# marker length of long side and short side
marker_size = params["pattern_size"]
marker_l = params["grid_length"] * marker_size[1]
marker_s = params["grid_length"] * marker_size[0]

not_segmented = True


# segment single frame of the point cloud into several segments
def seg_pcd(csv_path, save_folder_path):
    seg_num_thre = 3
    jdc_points_num_thres = 0
    seg_count = 0
    jdc_collection = jdc_segments_collection()
    jdc_collection.set_csv_path(csv_path)
    print ("csv_file loaded!")
    jdc_collection.get_potential_segments()

    clustered_seg_list = jdc_collection.cluster_seg()
    potential_seg_co_list = list()
    for tmp_seg_co in clustered_seg_list:
        # if is_human(tmp_seg_co):
        # color_tuple=np.array([0,255,0])
        potential_seg_co_list.append(tmp_seg_co)
    twice_clustered_seg_list = clustered_seg_list

    print ("twice_clustered_seg num=" + str(len(twice_clustered_seg_list)))

    parts = csv_path.split("/")
    if os.path.isdir(save_folder_path + parts[-1].split(".")[0]):
        shutil.rmtree(save_folder_path + parts[-1].split(".")[0])
    os.makedirs(save_folder_path + parts[-1].split(".")[0])

    count_after_filter = 0
    for tmp_seg_co in twice_clustered_seg_list:
        if len(tmp_seg_co) > seg_num_thre:
            count_after_filter += 1
            list_for_pedestrians_pcd = list()
            list_for_jdcs = list()
            for j in range(len(tmp_seg_co)):
                tmp_seg = tmp_seg_co[j]
                list_for_jdcs.append(tmp_seg.points_xyz.tolist())
                for k in range(tmp_seg.points_xyz.shape[0]):
                    point = tmp_seg.points_xyz[k, :]
                    list_for_pedestrians_pcd.append(point)
            arr_for_pedestrians_pcd = np.asarray(list_for_pedestrians_pcd, dtype=np.float32)
            if arr_for_pedestrians_pcd.shape[0] > 0:
                pcd_pedestrian = pcl.PointCloud(arr_for_pedestrians_pcd)

                parts = csv_path.split("/")

                if pcd_pedestrian.size > jdc_points_num_thres:
                    save_path_for_pedestrian_txt = save_folder_path + "/" + parts[-1].split(".")[0] + "/" + \
                                                   parts[-1].split(".")[0] + "block" + str(
                        seg_count) + ".txt"
                    seg_count += 1
                    cPickle.dump(list_for_jdcs, open(save_path_for_pedestrian_txt, 'wb'))
            del arr_for_pedestrians_pcd
            del list_for_pedestrians_pcd
            del list_for_jdcs
    del jdc_collection


# utilize the defined functions to get the chessboard's corners for single frame point cloud
def run(csv_path, save_folder_path=os.path.join(params['base_dir'], "output/pcd_seg/"), size=marker_size):
    if not_segmented:
        seg_pcd(csv_path, save_folder_path)

    '''parts = csv_path.split("/")
    find_marker_path = save_folder_path + parts[-1].split(".")[0] + "/"

    marker_pkl = find_marker(file_path=os.path.abspath(find_marker_path) + "/", csv_path=csv_path)
    marker_full_data_arr = exact_full_marker_data(csv_path, marker_pkl)

    # fit the points to the plane model
    model = get_plane_model(marker_full_data_arr[:, :3])
    pl_p = np.array([0, 0, -model[3] / model[2]])  # a point on the plane of the model
    normal = np.array(model[:3])
    fitted_list = []
    for i in marker_full_data_arr[:, :3]:
        b = p2pl_proj(normal, pl_p, i)
        fitted_list.append(b)
    marker_data_arr_fitted = np.array(fitted_list)
    marker_full_data_arr_fitted = np.hstack([marker_data_arr_fitted, marker_full_data_arr[:, 3:]])

    # trans chessboard
    if 1:
        # render for model of checkerboard
        rot1, transed_pcd = transfer_by_pca(marker_data_arr_fitted)
        t1 = transed_pcd.mean(axis=0)
        transed_pcd = transed_pcd - t1

    # calculate the rotate angle in xoy palne around the z axis
    if 1:
        low_intes, high_intens = get_gray_thre(marker_full_data_arr_fitted[:, params['intensity_col_ind']])
        print ("low_intes,high_intes:", low_intes, high_intens)
        rate = 2
        gray_zone = np.array([((rate - 1) * low_intes + high_intens), (low_intes + (rate - 1) * high_intens)]) / rate

        methods = ['Powell']
        res_dict = {}

        # for parallel processing
        args = (transed_pcd, marker_full_data_arr, gray_zone,)
        param_ls = [[method, args] for method in methods]
        res_ls = map(opt_min, param_ls)
        for item in res_ls:
            if item is not None:
                res_dict[item[0]] = item[1]

        res = res_dict[min(res_dict)][1]

        print (res_dict[min(res_dict)][0])
        print (res)

        rot2 = transforms3d.axangles.axangle2mat([0, 0, 1], res.x[0])
        t2 = np.array([res.x[1], res.x[2], 0])

        if 1:
            transed_pcd = np.dot(transforms3d.axangles.axangle2mat([0, 0, 1], res.x[0]),
                                 (transed_pcd + np.array([[res.x[1], res.x[2], 0]])).T).T

        gird_coords = generate_grid_coords()
        grid_ls = [(p[0]).flatten()[:2] for p in gird_coords]
        corner_arr = np.transpose(np.array(grid_ls).reshape(size[0], size[1], 2)[1:, 1:], (1, 0, 2))

    return [rot1, t1, rot2, t2, corner_arr, res.x, os.path.relpath(marker_pkl[0])]'''


def main_for_pool(i):
    pcd_file = os.path.join(params['base_dir'], "pcd/") + str(i).zfill(params["file_name_digits"]) + ".csv"
    print (pcd_file)
    try:
        result = run(csv_path=pcd_file)
        print (result)

    except AssertionError:
        print("marker cannot be found")
        print("skip " + pcd_file)

if __name__ == '__main__':
    print('Main')
    #main_for_pool(1)

