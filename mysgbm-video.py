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
array_datal = np.array([[562.033986323064, -1.20112458901875, 368.580858376389],
                        [0, 562.529949757492, 303.887885031138],
                        [0, 0, 1]])
array_datar = np.array([[572.929758405986, -3.74046455734938, 365.816495202695],
                        [0, 572.333398525109, 315.108984903095],
                        [0, 0, 1]])

left_camera_matrix = array_datal.T
right_camera_matrix = array_datar.T

# 畸变系数,K1、K2、K3为径向畸变,P1、P2为切向畸变
left_distortion = np.array([[0.124785954085824, -0.108200579559554, 0.00139760804551312, -0.00545030823545750, 0.001]])
right_distortion = np.array([[0.0827221706506679,0.0936542340045818, 0.00862240665117579,-0.00249349828386763, 0.001]])

# left_distortion = np.array([[0, 0, 0, 0, 0]])
# right_distortion = np.array([[0, 0, 0, 0, 0]])

# 旋转矩阵
R = np.array([[0.994844339868849, -0.0345781648731146, -0.0953367187652339],
                [0.0327011285263699, 0.999240701327606, -0.0211815250493617],
                [0.0959967479866658, 0.0179547020115234, 0.995219700896070]])
# 平移矩阵
T = np.array([142.208416260587, -0.109666878255351, 9.44528784942127])

size = (640, 480)

R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                  right_camera_matrix, right_distortion, size, R,
                                                                  T)

# 校正查找映射表,将原始图像和校正后的图像上的点一一对应起来
left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)
print(Q)

# --------------------------鼠标回调函数---------------------------------------------------------
#   event               鼠标事件
#   param               输入参数
# -----------------------------------------------------------------------------------------------
def onmouse_pick_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        threeD = param
        print('\n像素坐标 x = %d, y = %d' % (x, y))
        # print("世界坐标是：", threeD[y][x][0], threeD[y][x][1], threeD[y][x][2], "mm")
        print("世界坐标xyz 是：", threeD[y][x][0] / 1000.0, threeD[y][x][1] / 1000.0, threeD[y][x][2] / 1000.0, "m")

        distance = math.sqrt(threeD[y][x][0] ** 2 + threeD[y][x][1] ** 2 + threeD[y][x][2] ** 2)
        distance = distance / 1000.0  # mm -> m
        print("距离是：", distance, "m")

# 打开摄像头
capl = cv2.VideoCapture(1)
capl.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capl.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

capr = cv2.VideoCapture(2)
capr.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capr.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

WIN_NAME = 'Deep disp'
cv2.namedWindow(WIN_NAME, cv2.WINDOW_AUTOSIZE)

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
        # 计算视差
        disparity = stereo.compute(img1_rectified, img2_rectified)

        # 归一化函数算法，生成深度图（灰度图）
        disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # 生成深度图（颜色图）
        dis_color = disparity
        dis_color = cv2.normalize(dis_color, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        dis_color = cv2.applyColorMap(dis_color, 2)

        # 计算三维坐标数据值
        threeD = cv2.reprojectImageTo3D(disparity, Q, handleMissingValues=True)
        # 计算出的threeD，需要乘以16，才等于现实中的距离
        threeD = threeD * 16

        cv2.imshow("depth", dis_color)
        cv2.imshow(WIN_NAME, disp)  # 显示深度图的双目画面
        cv2.imshow("img1_rectified", img1_rectified)

        # 获取左右相机中目标点的三维坐标值
        point_left = threeD[int(cly), int(clx)]
        point_right = threeD[int(cry), int(crx)]

        # 计算目标点的深度值（欧氏距离）- 这里简单地计算左右相机中对应点的距离
        depth = math.sqrt((point_left[0] - point_right[0]) ** 2 + (point_left[1] - point_right[1]) ** 2 + (point_left[2] - point_right[2]) ** 2)
        print("depth:", depth)

        # 鼠标回调事件
        cv2.setMouseCallback("depth", onmouse_pick_points, threeD)

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