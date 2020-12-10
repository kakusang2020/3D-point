import numpy as np
import open3d as o3d
import random
import os
from sklearn import cluster, datasets, mixture
import queue
from scipy.spatial import KDTree, cKDTree


def get_plane(points):
    # ax + by + cz + d = 0, set d=1
    A = np.hstack((points, np.ones((points.shape[0], 1), dtype=np.float64)))
    AtA = np.matmul(A.T, A)
    u, s, vh = np.linalg.svd(AtA)
    vector = u[:, -1] / np.linalg.norm(u[:, -1])
    return float(vector[0]), float(vector[1]), float(vector[2]), float(vector[3])


def get_inner_idx(points, plane, thres_hold):
    A, B, C, D = plane
    diss_all = np.abs(A * points[:, 0] + B * points[:, 1] + C * points[:, 2] + D) / np.sqrt(A * A + B * B + C * C)
    return diss_all < thres_hold


def get_inner(points, plane, thres_hold):
    return points[get_inner_idx(points, plane, thres_hold)]


def get_vote(points, plane, thres_hold):
    inner = get_inner(points, plane, thres_hold)
    vote = int(inner.shape[0])
    return vote


def clustuer_points(points):
    min_distance = 1.0
    min_samples = 10

    #colors = np.zeros(points.shape)
    #point_cloud = o3d.geometry.PointCloud()
    #point_cloud.points = o3d.utility.Vector3dVector(points)

    reserved_points = points
    label = np.zeros(reserved_points.shape[0], dtype=np.int)
    sci_kdtree = cKDTree(reserved_points)

    lable_idx = 1
    reserved_idx = set(list(range(0, points.shape[0])))
    while (len(reserved_idx) > 0):
        # 找到第一个未visited的点
        cluster_idx = reserved_idx.pop()
        cluster_point = points[cluster_idx, :]
        neighbors_indices = sci_kdtree.query_ball_point(cluster_point, min_distance)
        my_lable = lable_idx

        if len(neighbors_indices) >= min_samples:
            label[cluster_idx] = lable_idx
            lable_idx += 1
        else:
            continue

        q = queue.Queue()
        for item in neighbors_indices:
            if (item in reserved_idx):
                reserved_idx.remove(item)
                label[item] = my_lable
                q.put(item)

        while not q.empty():
            front = q.get()
            extend_point = points[front, :]
            extend_neighbor = sci_kdtree.query_ball_point(extend_point, min_distance)
            for one_nei in extend_neighbor:
                if (one_nei in reserved_idx):
                    label[one_nei] = my_lable
                    reserved_idx.remove(one_nei)
                    q.put(one_nei)
        if (len(neighbors_indices) >= min_samples):
            print('new label:', my_lable, 'with', np.sum(label == my_lable), 'points')
            #colors = np.zeros(points.shape)
            #colors[label == my_lable, 0] = 1
            #point_cloud.colors = o3d.utility.Vector3dVector(colors)

    for ii in range(len(label)):
        if (label[ii] == -1):
            num_neig = len(sci_kdtree.query_ball_point(points[ii], min_distance))
            print('idx:', ii, 'has', num_neig, 'neighbors')
            assert (num_neig < min_samples)

    return label


def get_ground(points):
    # ransac
    num_points = int(points.shape[0])
    p = 0.99  # 算法有效率
    w = 0.6  # 内点率
    model_num = 5  # 模型所需点的数量
    k = int(np.log(1 - p) / np.log(1 - w ** model_num))
    inner_distance_thres = 0.5

    max_vote = 0
    max_candidate = None
    for ii in range(k):
        candidate = set()
        while (len(candidate) < model_num):
            one_idx = random.randint(0, num_points - 1)
            candidate.add(one_idx)
        plane = get_plane(points[list(candidate), :])
        vote = get_vote(points, plane, inner_distance_thres)
        if (vote > max_vote):
            max_vote = vote
            max_candidate = candidate

    max_plane = get_plane(points[list(max_candidate), :])

    final_plane = get_plane(get_inner(points, max_plane, inner_distance_thres))

    return get_inner_idx(points, final_plane, inner_distance_thres), final_plane


def process_one_file(file_path):
    flatened = np.fromfile(file_path, dtype=np.float32)
    data = flatened.reshape((-1, 4))

    point_cloud = o3d.geometry.PointCloud()
    points = data[:, 0:3].astype(np.float64)
    point_cloud.points = o3d.utility.Vector3dVector(points)

    if (False):
        point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    ground_idx, max_plane = get_ground(points)
    colors = np.zeros(points.shape)
    colors[ground_idx, 2] = 1

    clustered_label = clustuer_points(points[np.logical_not(ground_idx)])
    all_label = np.zeros(points.shape[0], dtype=np.int)
    all_label[np.logical_not(ground_idx)] = clustered_label
    clustered_label = all_label

    colors_all = np.random.rand(points.shape[0], 3)
    for ii in range(points.shape[0]):
        if clustered_label[ii] != 0:
            colors[ii] = colors_all[clustered_label[ii], :]
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([point_cloud],window_name=file_path.split('/')[-1])  # 显示原始点云

    pass


if __name__ == '__main__':
    root_dir = '/home/libaoyu/HDMnt/深蓝学院/三维点云处理/第4章/kitti_point_clouds'
    print(o3d.__version__)
    for file_name in sorted(os.listdir(root_dir)):
        full_path = os.path.join(root_dir, file_name)
        process_one_file(full_path)
