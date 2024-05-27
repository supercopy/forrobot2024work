import cv2
import numpy as np
import time

kernel = np.ones((5,5),np.uint8)
kernel_open = np.ones((10,10),np.uint8)

capl = cv2.VideoCapture(1)#左相机
capl.set(3, 640) #设置分辨率宽
capl.set(4, 480) #设置分辨率高

capr = cv2.VideoCapture(2)#右相机
capr.set(3, 640) #设置分辨率宽
capr.set(4, 480) #设置分辨率高

#相机标定参数：
array_datal = np.array([[562.033986323064, -1.20112458901875, 368.580858376389],
                        [0, 562.529949757492, 303.887885031138],
                        [0, 0, 1]])
array_datar = np.array([[572.929758405986, -3.74046455734938, 365.816495202695],
                        [0, 572.333398525109, 315.108984903095],
                        [0, 0, 1]])

#相机畸变参数：
left_distortion = np.array([[0.124785954085824, -0.108200579559554, 0.00139760804551312, -0.00545030823545750, 0.001]])
right_distortion = np.array([[0.0827221706506679,0.0936542340045818, 0.00862240665117579,-0.00249349828386763, 0.001]])

#相机旋转矩阵：
R = np.array([[0.994844339868849, -0.0345781648731146, -0.0953367187652339],
                [0.0327011285263699, 0.999240701327606, -0.0211815250493617],
                [0.0959967479866658, 0.0179547020115234, 0.995219700896070]])

# 平移矩阵
T = np.array([142.208416260587, -0.109666878255351, 9.44528784942127])

hsv_low = np.array([13, 105, 83])
hsv_high = np.array([31, 255, 255])

clX = 1
clY = 1

#图像处理并返回轮廓点
def get_contours(image):
    image = cv2.GaussianBlur(image, (5,5), 0) #高斯模糊
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) #转换为HSV
    image_dst = cv2.inRange(image_hsv, hsv_low, hsv_high) #二值化
    image_open = cv2.morphologyEx(image_dst, cv2.MORPH_OPEN, kernel_open) #开运算
    image_dilated = cv2.dilate(image_open, kernel, iterations=1) #膨胀操作
    contours, hierarchy = cv2.findContours(image_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #查找轮廓
    return image_dilated, contours

#求最小外接矩形并返回顶点，使用该函数在遍历轮廓内
def get_rectangle(cnt):
    rect = cv2.minAreaRect(cnt) #求最小外接矩形
    box = cv2.boxPoints(rect) #返回矩形的四个顶点
    box = np.int0(box) #转换为整型
    return box

#通过矩值计算中心点
def get_center(M):
    centerX = int(M["m10"] / M["m00"])
    centerY = int(M["m01"] / M["m00"])
    return centerX, centerY

while(True):
    retl, framel = capl.read()
    imagel = framel.copy()

    retr, framer = capr.read()
    imager = framer.copy()

    # 处理图像，得到轮廓
    imagel_dilated, contoursl = get_contours(imagel)
    imager_dilated, contoursr = get_contours(imager)

    for cntl in contoursl:
        areal = cv2.contourArea(cntl)
        if areal < 100:
            continue
        boxl = get_rectangle(cntl)  # 求最小外接矩形
        cv2.drawContours(imagel, [boxl], 0, (255, 0, 0), 2)  # 画出矩形
        Ml = cv2.moments(cntl)  # 计算矩值，方便得到矩形中心点

        if Ml["m00"] != 0:
            reglX = clX
            reglY = clY
            clX, clY = get_center(Ml)  # 得到矩形中心点
            cv2.circle(imagel, (clX, clY), 5, (0, 0, 255), -1)  # 画出矩形中心点
            if clX != reglX or clY != reglY:
                print(clX, clY)

    for cntr in contoursr:
        arear = cv2.contourArea(cntr)
        if arear < 100:
            continue
        boxr = get_rectangle(cntr)  # 求最小外接矩形
        cv2.drawContours(imager, [boxr], 0, (255, 0, 0), 2)  # 画出矩形
        Mr = cv2.moments(cntr)  # 计算矩值，方便得到矩形中心点
        
        if Mr["m00"] != 0:
            regrX = crX
            regrY = crY
            crX, crY = get_center(Mr)  # 得到矩形中心点
            cv2.circle(imager, (crX, crY), 5, (0, 0, 255), -1)  # 画出矩形中心点
            if crX != regrX or crY != regrY:
                print(crX, crY)

    cv2.imshow('imagel', imagel)
    cv2.imshow('imager', imager)
    cv2.imshow('imagel_dilated', imagel_dilated)
    cv2.imshow('imager_dilated', imager_dilated)

    time.sleep(0.1)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
capl.release()
capr.release()
cv2.destroyAllWindows()