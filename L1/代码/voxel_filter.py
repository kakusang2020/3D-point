# 实现voxel滤波，并加载数据集中的文件进行验证

import open3d as o3d 
import os
import numpy as np
from pyntcloud import PyntCloud

# 功能：对点云进行voxel滤波
# 输入：
#     point_cloud：输入点云 (Nx3)
#     leaf_size: voxel尺寸
def voxel_filter(point_cloud, leaf_size,if_mean=False):
    filtered_points = []
    # 作业3
    # 屏蔽开始
    x_min,x_max,y_min,y_max,z_min,z_max = np.min(point_cloud[:,0]),np.max(point_cloud[:,0]), \
    np.min(point_cloud[:,1]),np.max(point_cloud[:,1]),np.min(point_cloud[:,2]),np.max(point_cloud[:,2])

    Dx,Dy,Dz = (x_max-x_min)/leaf_size,(y_max-y_min)/leaf_size,(z_max-z_min)/leaf_size

    min_vec = np.array([x_min,y_min,z_min])
    index = np.floor((point_cloud.copy()-min_vec)/leaf_size)
    h_index = index[:,0]+index[:,1]*Dx+index[:,2]*Dx*Dy

    for index in np.unique(h_index):
        point_choosed = point_cloud[h_index==index]
        if if_mean:
            filtered_points.append(np.mean(point_choosed,axis=0))
        else:
            filtered_points.append(point_choosed[np.random.choice(a=point_choosed.shape[0])])
    # 屏蔽结束

    # 把点云格式改成array，并对外返回
    filtered_points = np.array(filtered_points, dtype=np.float64)
    return filtered_points

def main():
    # 指定点云路径
    path = '../../../data/modelnet40_normal_resampled/'

    shape_name_list = np.loadtxt(os.path.join(path,'modelnet40_shape_names.txt') if os.path.isdir(path) else None,dtype=str)
    pc_list = []

    for item in shape_name_list:
        filename = os.path.join(path,item,item+'_0001.txt')
        pointcloud = np.loadtxt(filename,delimiter=',')[:,0:3]
        pointcloud_o3d = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pointcloud))
        filtered_cloud = voxel_filter(pointcloud,0.1,if_mean=False)+1*np.array([0,1,0])
        filteredcloud_o3d = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(filtered_cloud))

        o3d.visualization.draw_geometries([pointcloud_o3d,filteredcloud_o3d])
    
if __name__ == '__main__':
    main()
