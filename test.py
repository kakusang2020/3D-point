import open3d as o3d
import numpy as np
 
 
points = np.random.rand(10000, 3)
point_cloud = o3d.PointCloud()
point_cloud.points = o3d.Vector3dVector(points)
o3d.draw_geometries([point_cloud])