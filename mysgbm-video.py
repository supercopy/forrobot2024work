import cv2
import numpy as np
import time
import random
import math

from orangeblock import fruit_center

# -----------------------------------双目相机的基本参数---------------------------------------------------------
#   left_camera_matrix          左相机的内参矩阵
#   right_camera_matrix         右相机的内参矩阵
#
#   left_distortion             左相机的畸变系数    格式(K1,K2,P1,P2,0)
#   right_distortion            右相机的畸变系数
# -------------------------------------------------------------------------------------------------------------
# 左镜头的内参，如焦距
array_datal = np.array([[665.708680641050, 0, 608.007556651546],
                 [0, 658.259575109014, 349.414700641615],
                 [0, 0, 1]])
array_datar = np.array([[658.816842107741, 0, 628.024134809292],
                 [0, 651.968299224625, 346.395327320762],
                 [0, 0, 1]])

left_camera_matrix = array_datal
right_camera_matrix = array_datar

# left_camera_matrix = array_datal.T
# right_camera_matrix = array_datar.T

# 相机的焦距（单位：像素）
focal_length = (1280 * 35 / 36) / 1000 #使用35mm相机的焦距，从exif信息中获取，35mm相机的胶片尺寸是36mm*24mm，图像为1280*720，转换单位为m

scale_factor = 4  # 深度图转换的比例因子，可以根据实际情况进行调整

# 畸变系数,K1、K2、K3为径向畸变,P1、P2为切向畸变,次序为k1,k2,p1,p2,k3,这份代码无k3参数，在reg2中有带k3参数的代码
left_distortion = np.array([[0.0855575067223019, -0.101075323042139, 0.000717078106918775, -0.00379148552763136, 0]])
right_distortion = np.array([[0.0798755813002228, -0.0953198903280210, 0.000428987925347287, 0.00180737891825946, 0]])

# left_distortion = np.array([[0, 0, 0, 0, 0]])
# right_distortion = np.array([[0, 0, 0, 0, 0]])

# 旋转矩阵
R = np.array([[0.998223206868815, 0.0499560086850059, -0.0324780920745370],
                 [-0.0498410539286819, 0.998747732943592, 0.00433996351528645],
                 [0.0326542280848393, -0.00271350995933524, 0.999463024954842]])
# 平移矩阵
T = np.array([-142.347426331190, 5.34298687576439, -1.56832656665621])

size = (1280, 720)

#R1、R2、P1、P2是校正后的相机内参矩阵，Q是立体校正后的图像的视差图（深度图），validPixROI1和validPixROI2是校正后的图像的有效像素区域
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                  right_camera_matrix, right_distortion, size, R,
                                                                  T)

# 校正查找映射表,将原始图像和校正后的图像上的点一一对应起来
#left_map1和left_map2，分别表示左相机校正查找映射表的x和y坐标，right_map1和right_map2分别表示右相机校正查找映射表的x和y坐标
left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)

# 根据相机内参计算实际距离的转换函数
def convert_depth_value(depth_value, focal_length, scale_factor):
    return depth_value * focal_length / scale_factor

# --------------------------鼠标回调函数---------------------------------------------------------
#   event               鼠标事件
#   param               输入参数
# -----------------------------------------------------------------------------------------------
def onmouse_pick_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        threeD_left = param
        print('\n像素坐标 x = %d, y = %d' % (x, y))
        # print("世界坐标是：", threeD_left[y][x][0], threeD_left[y][x][1], threeD_left[y][x][2], "mm")
        print("世界坐标xyz 是：", threeD_left[y][x][0] / 1000.0, threeD_left[y][x][1] / 1000.0, threeD_left[y][x][2] / 1000.0, "m")

        distance = math.sqrt(threeD_left[y][x][0] ** 2 + threeD_left[y][x][1] ** 2 + threeD_left[y][x][2] ** 2)
        distance = distance / 1000.0  # mm -> m
        real_distance = convert_depth_value(distance, focal_length, scale_factor)
        print("真实距离是:", real_distance, "m")

# 打开摄像头
capl = cv2.VideoCapture(2)
capl.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
capl.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

capr = cv2.VideoCapture(1)
capr.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
capr.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 读取视频
fps = 0.0
retl, framel = capl.read()
retr, framer = capr.read()

if __name__ == "__main__":
    while retl or retr:
        stereo_vision = fruit_center(framel, framer)
        clx, cly, crx, cry = stereo_vision.detect_objects() #得到目标点坐标
        # 开始计时
        t1 = time.time()
        # 是否读取到了帧，读取到了则为True
        retl, framel = capl.read()
        retr, framer = capr.read()

        # 确保两张图像形状相同
        if framel.shape == framer.shape:
        # 将两个图像叠加在一起
            frames = np.hstack((framel, framer))
        # 将BGR格式转换成灰度图片，用于畸变矫正
        imgL = cv2.cvtColor(framel, cv2.COLOR_BGR2GRAY)
        imgR = cv2.cvtColor(framer, cv2.COLOR_BGR2GRAY)

        # 重映射，就是把一幅图像中某位置的像素放置到另一个图片指定位置的过程。
        # 依据MATLAB测量数据重建无畸变图片,输入图片要求为灰度图
        img1_rectified = cv2.remap(imgL, left_map1, left_map2, cv2.INTER_LINEAR)
        img2_rectified = cv2.remap(imgR, right_map1, right_map2, cv2.INTER_LINEAR)

        # 转换为opencv的BGR格式
        imageL = cv2.cvtColor(img1_rectified, cv2.COLOR_GRAY2BGR)
        imageR = cv2.cvtColor(img2_rectified, cv2.COLOR_GRAY2BGR)

        # ------------------------------------SGBM算法----------------------------------------------------------
        #   blockSize                   深度图成块，blocksize越低，其深度图就越零碎，0<blockSize<10
        #   img_channels                BGR图像的颜色通道，img_channels=3，不可更改
        #   numDisparities              SGBM感知的范围，越大生成的精度越好，速度越慢，需要被16整除，如numDisparities
        #                               取16、32、48、64等
        #   mode                        sgbm算法选择模式，以速度由快到慢为：STEREO_SGBM_MODE_SGBM_3WAY、
        #                               STEREO_SGBM_MODE_HH4、STEREO_SGBM_MODE_SGBM、STEREO_SGBM_MODE_HH。精度反之
        # ------------------------------------------------------------------------------------------------------
        blockSize = 5
        img_channels = 3
        stereo = cv2.StereoSGBM_create(minDisparity=1,
                                    numDisparities=128,
                                    blockSize=blockSize, #越小，深度图越稀碎
                                    P1=8 * img_channels * blockSize * blockSize,
                                    P2=32 * img_channels * blockSize * blockSize,
                                    disp12MaxDiff=-1,
                                    preFilterCap=1,
                                    uniquenessRatio=10,
                                    speckleWindowSize=100,
                                    speckleRange=100,
                                    mode=cv2.STEREO_SGBM_MODE_HH
                                    )
        # 计算左相机视差
        disparity_left = stereo.compute(img1_rectified, img2_rectified)
        # 归一化函数算法，生成深度图（灰度图）
        disp_left = cv2.normalize(disparity_left, disparity_left, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        #求右相机深度图
        disparity_right = stereo.compute(img2_rectified, img1_rectified)
        disp_right = cv2.normalize(disparity_right, disparity_right, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # 生成左相机深度图（颜色图）
        dis_color_left = disparity_left
        dis_color_left = cv2.normalize(dis_color_left, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        dis_color_left = cv2.applyColorMap(dis_color_left, 2)

        #生成右相机深度图（颜色图）
        dis_color_right = disparity_right
        dis_color_right = cv2.normalize(dis_color_right, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        dis_color_right = cv2.applyColorMap(dis_color_right, 2)
        
        # 计算三维坐标数据值
        threeD_left = cv2.reprojectImageTo3D(disparity_left, Q, handleMissingValues=True)
        threeD_right = cv2.reprojectImageTo3D(disparity_right, Q, handleMissingValues=True)

        # 计算出的threeD，需要乘以16，才等于现实中的距离
        threeD_left = threeD_left * 16
        threeD_right = threeD_right * 16

        # 获取左右相机中目标点的三维坐标值
        point_left = threeD_left[int(cly), int(clx)]
        point_right = threeD_right[int(cry), int(crx)]

        # 计算目标点的深度值（欧氏距离）- 这里简单地计算左右相机中对应点的距离
        depth = math.sqrt((point_left[0] - point_right[0]) ** 2 + (point_left[1] - point_right[1]) ** 2 + (point_left[2] - point_right[2]) ** 2)
        # 转换深度值为实际世界距离
        real_distance = convert_depth_value(depth, focal_length, scale_factor)
        print("真实距离:", real_distance, "m")

        # cv2.imshow('Deep disp_left', disp_left)  # 显示深度图的双目画面
        # cv2.imshow('Deep disp_right', disp_right)
        cv2.imshow('disparity_left', dis_color_left)# 显示color深度图的双目画面
        cv2.imshow('disparity_right', dis_color_right)
        cv2.imshow("img1_rectified", img1_rectified)

        # 鼠标回调事件
        cv2.setMouseCallback("disparity_left", onmouse_pick_points, threeD_left)
        cv2.setMouseCallback("disparity_right", onmouse_pick_points, threeD_right)

        #完成计时，计算帧率
        fps = (fps + (1. / (time.time() - t1))) / 2
        frame = cv2.putText(framel, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 若键盘按下q则退出播放
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    # 释放资源
    capl.release()
    capr.release()

    # 关闭所有窗口
    cv2.destroyAllWindows()