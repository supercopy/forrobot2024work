from glob import glob
import numpy as np
import os
import cv2

imgs = [cv2.imread(i, cv2.IMREAD_GRAYSCALE) for i in glob(os.sep.join(['./pics/', '*.jpg']))] #将路径下所有jpg图片转换为灰度图存到imgs中
cImgSize = imgs[0].shape[::-1] # (w, h) #获取所有图像的宽度和高度
cBoardSize = (8, 5) # 棋盘格大小

cLen = len(imgs) # 获取图像数量
assert 0 == cLen%2 # 确保图像数量为偶数
cLen = cLen//2 # 获取图像数量的一半

def detectCorners(img): # 检测棋盘格角点
    ok, corners = cv2.findChessboardCorners(img, cBoardSize) # 检测棋盘格角点
    assert ok # 确保检测成功
    cv2.cornerSubPix(img, corners, (50, 50), (-1, -1), (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.01)) # 优化角点位置
    return corners

cImgPoints = [detectCorners(img).reshape(-1, 2) for img in imgs] # 获取角点坐标

cObjPoints = np.zeros((np.prod(cBoardSize), 3), np.float32) # 创建一个用于存储角点坐标的数组
cObjPoints[:, :2] = np.indices(cBoardSize).T.reshape(-1, 2) # 生成角点坐标
cObjPoints = cLen*[cObjPoints] # 重复生成角点坐标

K0 = cv2.initCameraMatrix2D(cObjPoints, cImgPoints[:cLen], cImgSize, 0) # 获取相机矩阵
K1 = cv2.initCameraMatrix2D(cObjPoints, cImgPoints[cLen:], cImgSize, 0) # 获取相机矩阵

rms, K0, D0, K1, D1, R, T, E, F = cv2.stereoCalibrate( # 获取立体相机标定参数
    cObjPoints, cImgPoints[:cLen], cImgPoints[cLen:], # 棋盘格角点坐标
    K0, None, K1, None, # 相机矩阵
    cImgSize, # 图像尺寸
    flags = cv2.CALIB_FIX_ASPECT_RATIO # 确保长宽比保持一致
            | cv2.CALIB_ZERO_TANGENT_DIST # 确保切向畸变系数为0
            | cv2.CALIB_USE_INTRINSIC_GUESS # 确保使用初始相机矩阵
            | cv2.CALIB_SAME_FOCAL_LENGTH # 确保使用相同焦距
            | cv2.CALIB_RATIONAL_MODEL # 确保使用rational model
            | cv2.CALIB_FIX_K3 # 确保使用相同焦距
            | cv2.CALIB_FIX_K4 # 确保使用相同焦距
            | cv2.CALIB_FIX_K5, # 确保使用相同焦距
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-5) # 终止条件
)

R0, R1, P0, P1, Q, roi0, roi1 = cv2.stereoRectify(K0, D0, K1, D1, cImgSize, R, T, flags = cv2.CALIB_ZERO_DISPARITY, alpha = 0) # 获取立体校正参数

def rectify(img, K, D, R, P): # 校正图像
    m0, m1 = cv2.initUndistortRectifyMap(K, D, R, P, cImgSize, cv2.CV_16SC2) # 获取校正映射
    return cv2.remap(img, m0, m1, cv2.INTER_LINEAR) # 校正图像

melonL = rectify(cv2.imread('cornerl_1.jpg'), K0, D0, R0, P0) # 校正图像
melonR = rectify(cv2.imread('cornerr_1.jpg'), K1, D1, R1, P1) # 校正图像

melon = np.concatenate((melonL, melonR), axis = 1) # 拼接图像
melon[::40, :] = (0, 255, 0) # 设置特定区域的像素值 # 设置特定区域的像素值

def cb(e, x, y, f, p): # 鼠标回调函数
    global start # 全局变量
    global end # 全局变量
    if x < cv2.getTrackbarPos('disparities', 'depth')*16 or x > cImgSize[0]: # 排除超出范围的像素点
        return
    if e == cv2.EVENT_LBUTTONDOWN: # 鼠标左键按下
        start = points3d[y][x]
        print('起始:', (round(start[0],2),round(start[1],2), round(start[2],2))) # 打印起始点坐标
    if e == cv2.EVENT_LBUTTONUP:
        end = points3d[y][x] # 获取当前像素点的坐标
        print('终止:', (round(end[0],2),round(end[1],2),round(end[2],2))) # 打印终止点坐标
        distance = np.sqrt(np.sum((start - end)**2)) # 计算两点之间的距离
        print('距离:', distance, 'cm') # 打印两点之间的距离

cv2.namedWindow('depth')
cv2.namedWindow('rectify')
cv2.createTrackbar('disparities', 'depth', 5, 60, lambda x: None) # 创建滑动条
cv2.createTrackbar('block', 'depth', 3, 32, lambda x: None) # 创建滑动条
cv2.setMouseCallback('rectify', cb, None) # 设置鼠标回调函数

while True:
    d = cv2.getTrackbarPos('disparities', 'depth')*16 # 获取滑动条的位置
    b = cv2.getTrackbarPos('block', 'depth') # 获取滑动条的位置

    matcher0 = cv2.StereoSGBM_create(0, d, b, 24*b, 96*b, 12, 10, 50, 32, 63, cv2.STEREO_SGBM_MODE_SGBM_3WAY) # 创建StereoSGBM对象
    matcher1 = cv2.ximgproc.createRightMatcher(matcher0) # 创建matcher1对象
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher0) # 创建wls_filter对象
    wls_filter.setLambda(80000) # 设置lambda参数
    wls_filter.setSigmaColor(1.3) # 设置sigmaColor参数
    disp0 = np.int16(matcher0.compute(melonL, melonR)) # 计算视差图
    disp1 = np.int16(matcher1.compute(melonR, melonL)) # 计算视差图

    depth = wls_filter.filter(disp0, melonL, None, disp1).astype(np.float32)/16. # 计算深度图
    points3d = cv2.reprojectImageTo3D(depth, Q) # 计算3D点云

    frame = melon.copy() # 创建一个与原图像相同大小的空白图像
    cv2.line(frame, (d, 0), (d, cImgSize[1]), (0, 255, 0), 1) # 绘制一条垂直线
    cv2.imshow('rectify', frame) # 显示图像

    depthImg = depth.copy() # 创建一个与原图像相同大小的空白图像
    depthImg = np.uint8(cv2.normalize(depthImg, depthImg, 255, 0, cv2.NORM_MINMAX)) # 归一化图像
    cv2.imshow('depth', depthImg) # 显示图像

    cv2.waitKey(100)