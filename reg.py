import numpy as np

class HomographyMapper:
    def __init__(self, points_1, points_2):
        self.points_1 = np.array([np.concatenate((p, [1])) for p in points_1])
        self.points_2 = np.array([np.concatenate((p, [1])) for p in points_2])
        H = np.linalg.inv(self.points_1.T @ self.points_1) @ self.points_1.T @ self.points_2
        self.H_4x4 = np.pad(H, ((0, 1), (0, 0)), mode='constant', constant_values=0)
        self.H_4x4[3, 3] = 1

    def map_points_with_homography(self, points_3d):
        points_homogeneous = np.column_stack((points_3d, np.ones((points_3d.shape[0], 1))))
        mapped_points = np.dot(self.H_4x4, points_homogeneous)
        mapped_points_cartesian = mapped_points[:, :3] / mapped_points[:, 3][:, None]
        return mapped_points_cartesian

# 定义四个第一三维坐标系点
right_xyz1 = np.array([0.11820718383789063, 0.05533146667480469, 1.0028218383789063])
right_xyz2 = np.array([-0.01996559715270996, 0.09150011444091796, 0.94927001953125])
right_xyz3 = np.array([-0.19349075317382813, 0.07384275817871094, 0.9843123168945312])
right_xyz4 = np.array([-0.38008270263671873, 0.05715393829345703, 1.164743896484375])

# 定义四个第二三维坐标系点
right_4 = np.array([-0.21501827451228817, -0.31151167908915567, -0.06251055943497788])
right_3 = np.array([-0.09883309329815113, -0.4565774141020829, -0.07697371230214964])
right_2 = np.array([0.09122137521581827, -0.4566295945409391, -0.07724894630523474])
right_1 = np.array([0.2409934766406208, -0.2882685735221945, -0.04749755037280354])

# 创建HomographyMapper对象并进行测试
points_1 = np.array([right_xyz1, right_xyz2, right_xyz3, right_xyz4])
points_2 = np.array([right_1, right_2, right_3, right_4])
mapper = HomographyMapper(points_1, points_2)

print("Mapped Point in Second 3D Coordinate System:")
points_3d_to_map = np.array([right_xyz1])
mapped_points_3d = mapper.map_points_with_homography(points_3d_to_map)
print(mapped_points_3d)