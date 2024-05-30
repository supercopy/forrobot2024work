# maded on 2024/5/29 20:30
#我需要实现两个三维坐标系的单应性变换，这份代码输出可用的单应性矩阵，从相机坐标系变为tcp坐标系

'''
算法理解:如果我想求出两个三维坐标系的单应性变换，是需要四个不同点在两个坐标系的坐标，
并且将三维坐标补充一维的齐次坐标，而后将四个点的四组相对应的坐标构建两个点集，
使用点集去计算出两个三维坐标系间的单应性矩阵H，而后扩展3*3单应性矩阵为4*4，最后使用该矩阵进行两个坐标系间的映射,该映射是单向的.
'''
'''
单应性矩阵使用：
H*X1=X2，其中X1是第一个三维坐标系中的点，X2是第二个三维坐标系中的点，H是单应性矩阵
注意:由于单应性矩阵是单向的，所以X1到X2的映射是单向的，X2到X1的映射是单向的，不能实现双向的映射
注意:由于点x1=(x1,y1,z1)的坐标添加了齐次坐标得到 x'1=(x1', y1', z1', 1)，求出的单应性矩阵是4*4的，齐次坐标点x1*H后，
得到的是映射后的其次坐标点x'2=(x'2,y'2,z'2,w'2)，所以需要将映射后得到的齐次坐标点x'2转换为非齐次坐标点x2=(x2, y2, z2)，
即除以x'2的第四个分量w'2，得到x2=(x'2/w'2, y'2/w'2, z'2/w'2)
'''
import cv2
import numpy as np

# 已知的四个点在第一三维坐标系
x1, y1, z1 = 1, 2, 3
x2, y2, z2 = 4, 5, 6
x3, y3, z3 = 7, 8, 9
x4, y4, z4 = 10, 11, 12
a1 = np.array([x1, y1, z1])
b1 = np.array([x2, y2, z2])
c1 = np.array([x3, y3, z3])
d1 = np.array([x4, y4, z4])

# 已知的四个点在第二三维坐标系
x1_2, y1_2, z1_2 = 0.5, 1.5, 2.5
x2_2, y2_2, z2_2 = 3.5, 4.5, 5.5
x3_2, y3_2, z3_2 = 6.5, 7.5, 8.5
x4_2, y4_2, z4_2 = 9.5, 10.5, 11.5
a2 = np.array([x1_2, y1_2, z1_2])
b2 = np.array([x2_2, y2_2, z2_2])
c2 = np.array([x3_2, y3_2, z3_2])
d2 = np.array([x4_2, y4_2, z4_2])

# 构建两个点集
#np.append(a1, 1)在数组末尾添加元素值1
#np.concatenate((a1, [1]))将数组a1与数组[1]连接起来
points_1 = np.array([np.concatenate((a1, [1])),
                     np.concatenate((b1, [1])),
                     np.concatenate((c1, [1])),
                     np.concatenate((d1, [1]))
                     ])

points_2 = np.array([np.concatenate((a2, [1])),
                     np.concatenate((b2, [1])),
                     np.concatenate((c2, [1])),
                     np.concatenate((d2, [1]))
                    ])

# 计算单应性矩阵，不考虑齐次维度
H, _ = cv2.findHomography(points_1[:, :3], points_2[:, :3])

# 将3x3的单应性矩阵扩展为4x4
H_4x4 = np.pad(H, ((0, 1), (0, 1)), mode='constant', constant_values=0)
H_4x4[3, 3] = 1

# 使用单应性矩阵将第一三维坐标系中的其他点映射到第二三维坐标系
def map_points_with_homography(points_3d, homography):
    points_homogeneous = np.column_stack((points_3d, np.ones((points_3d.shape[0], 1))))
    print(points_homogeneous)
    mapped_points = np.dot(homography, points_homogeneous.T).T
    mapped_points_cartesian = mapped_points[:, :3] / mapped_points[:, 3][:, None]
    return mapped_points_cartesian

# 测试代码
points_3d_to_map = np.array([[4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])

mapped_points_3d_2 = map_points_with_homography(points_3d_to_map, H_4x4)

print("Homography Matrix:")
print(H_4x4)

print("Mapped Points in Second 3D Coordinate System:")
print(mapped_points_3d_2)