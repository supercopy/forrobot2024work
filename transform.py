from gettcpXYZ import gettcp
from XYZtransformations_block import HomographyMapper
import numpy as np
import cv2

#从右往左
# 世界坐标xyz 是： 0.11820718383789063 0.05533146667480469 1.0028218383789063 m
# 真实距离是: 0.898915076797998 m
# 世界坐标xyz 是： -0.01996559715270996 0.09150011444091796 0.94927001953125 m
# 真实距离是: 0.847892127307597 m
# 世界坐标xyz 是： -0.19349075317382813 0.07384275817871094 0.9843123168945312 m
# 真实距离是: 0.8941012340018054 m
# 世界坐标xyz 是： -0.38008270263671873 0.05715393829345703 1.164743896484375 m
# 真实距离是: 1.0902423435761295 m

#第一三维坐标系四个点
right_xyz1 = np.array([0.11820718383789063, 0.05533146667480469, 1.0028218383789063])
right_xyz2 = np.array([-0.01996559715270996, 0.09150011444091796, 0.94927001953125])
right_xyz3 = np.array([-0.19349075317382813, 0.07384275817871094, 0.9843123168945312])
right_xyz4 = np.array([-0.38008270263671873, 0.05715393829345703, 1.164743896484375])
#第二三维坐标系四个点
right_4 = np.array([-0.21501827451228817, -0.31151167908915567, -0.06251055943497788])
right_3 = np.array([-0.09883309329815113, -0.4565774141020829, -0.07697371230214964])
right_2 = np.array([0.09122137521581827, -0.4566295945409391, -0.07724894630523474])
right_1 = np.array([0.2409934766406208, -0.2882685735221945, -0.04749755037280354])

points_1 = np.array([right_xyz1, right_xyz2, right_xyz3, right_xyz4])
points_2 = np.array([right_1, right_2, right_3, right_4])
print("point_1:")
print(points_1)
print("point_2:")
print(points_2)

mapper = HomographyMapper(points_1, points_2)
points_3d_to_map = np.array([right_1])
mapped_points_3d_2 = mapper.map_points_with_homography(points_1)
print("mapped_points_3d_2:")
print(mapped_points_3d_2)
