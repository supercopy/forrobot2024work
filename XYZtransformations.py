import cv2
import numpy as np

# 已知的同一点在第一三维坐标系和第二三维坐标系中的坐标

# 已知的同一点在第一三维坐标系中的坐标
point_3d_1 = np.array([1, 2, 3])  # 三维坐标系1中的点

# 已知的同一点在第二三维坐标系中的坐标
point_3d_2 = np.array([0.5, 1.5, 2.5])  # 三维坐标系2中的点

# 构建其他3组对应的点集数据
points_1 = np.array([np.concatenate((point_3d_1, [1])),
                     [4, 5, 6, 1],
                     [7, 8, 9, 1],
                     [10, 11, 12, 1]])

points_2 = np.array([np.concatenate((point_3d_2, [1])),
                     [3.5, 4.5, 5.5, 1],
                     [6.5, 7.5, 8.5, 1],
                     [9.5, 10.5, 11.5, 1]])

# 计算单应性矩阵
H, _ = cv2.findHomography(points_1[:, :3], points_2[:, :3])

# 将3x3的单应性矩阵扩展为4x4
H_4x4 = np.pad(H, ((0, 1), (0, 1)), mode='constant', constant_values=0)
H_4x4[3, 3] = 1

# 使用单应性矩阵将第一三维坐标系中的其他点映射到第二三维坐标系
def map_points_with_homography(points_3d, homography):
    points_homogeneous = np.column_stack((points_3d, np.ones((points_3d.shape[0], 1))))
    mapped_points = np.dot(homography, points_homogeneous.T).T
    mapped_points_cartesian = mapped_points[:, :3] / mapped_points[:, 3][:, None]
    return mapped_points_cartesian

# 测试代码
points_3d_to_map = np.array([[4, 5, 6], [7, 8, 9], [10, 11, 12]])

mapped_points_3d_2 = map_points_with_homography(points_3d_to_map, H_4x4)

print("Homography Matrix:")
print(H_4x4)

print("Mapped Points in Second 3D Coordinate System:")
print(mapped_points_3d_2)