import cv2
import numpy as np

class HomographyMapper: 
    '''这个类用于'''
    def __init__(self, points_1, points_2):
        self.points_1 = np.array([np.concatenate((p, [1])) for p in points_1])
        self.points_2 = np.array([np.concatenate((p, [1])) for p in points_2])
        H, _ = cv2.findHomography(self.points_1[:, :3], self.points_2[:, :3])
        self.H_4x4 = np.pad(H, ((0, 1), (0, 1)), mode='constant', constant_values=0)
        self.H_4x4[3, 3] = 1

    def map_points_with_homography(self, points_3d):
        points_homogeneous = np.column_stack((points_3d, np.ones((points_3d.shape[0], 1))))
        mapped_points = np.dot(self.H_4x4, points_homogeneous.T).T
        mapped_points_cartesian = mapped_points[:, :3] / mapped_points[:, 3][:, None]
        return mapped_points_cartesian

# 测试代码
# 已知的四个点在第一三维坐标系
points_1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
# 已知的四个点在第二三维坐标系
points_2 = [[0.5, 1.5, 2.5], [3.5, 4.5, 5.5], [6.5, 7.5, 8.5], [9.5, 10.5, 11.5]]

mapper = HomographyMapper(points_1, points_2)
points_3d_to_map = np.array([[13, 14, 15], [16, 17, 18], [19, 20, 21]])

mapped_points_3d_2 = mapper.map_points_with_homography(points_3d_to_map)

print("Mapped Points in Second 3D Coordinate System:")
print(mapped_points_3d_2)