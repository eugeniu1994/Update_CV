import os
import yaml

def get_default_params():
    path = '/src/Camera_Lidar/DATA/config.yaml'
    default_params_yaml = open(path, "r")
    if (int(yaml.__version__[0]) >= 5):
        params = yaml.safe_load(default_params_yaml)
    else:
        params = yaml.load(default_params_yaml)

    params['image_format'] = get_img_format(params['base_dir'])
    return params

def get_img_format(base_dir):
    file_ls = os.listdir(os.path.join(base_dir, "img"))
    for file in file_ls:
        ext = file.split(".")[-1]
        if ext in ["png", "PNG", "jpg", "JPG"]:
            return ext

# print get_default_params()

'''import numpy as np
import pcl

p = pcl.PointCloud(10)  # "empty" point cloud
a = np.asarray(p)  # NumPy view on the cloud
a[:] = 0  # fill with zeros
print(p[3])  # prints (0.0, 0.0, 0.0)
a[:, 0] = 1  # set x coordinates to 1
print(p[3])  # prints (1.0, 0.0, 0.0)

print('Done')'''

