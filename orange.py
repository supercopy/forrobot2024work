import cv2
import numpy as np
import time

kernel = np.ones((5,5),np.uint8)
kernel_open = np.ones((10,10),np.uint8)

cap = cv2.VideoCapture(1)#左相机
cap.set(3, 640) #设置分辨率宽
cap.set(4, 480) #设置分辨率高

# cap = cv2.VideoCapture(2)#右相机
# cap.set(3, 640) #设置分辨率宽
# cap.set(4, 480) #设置分辨率高

hsv_low = np.array([13, 105, 83])
hsv_high = np.array([31, 255, 255])

clX = 1
clY = 1

while(True):
    ret, frame = cap.read()
    imagel = frame.copy()

    #高斯模糊
    imagel = cv2.GaussianBlur(imagel, (5,5), 0)
    #转换为HSV
    imagel_hsv = cv2.cvtColor(imagel, cv2.COLOR_BGR2HSV)
    #二值化
    imagel_dst = dst = cv2.inRange(imagel_hsv, hsv_low, hsv_high)
    #开运算
    imagel_open = cv2.morphologyEx(imagel_dst, cv2.MORPH_OPEN, kernel_open)
    #膨胀操作
    imagel_dilated = cv2.dilate(imagel_open, kernel, iterations=1)
    #查找轮廓
    contours, hierarchy = cv2.findContours(imagel_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # for cnt in contours:
    #     #计算轮廓的面积
    #     area = cv2.contourArea(cnt)
    #     #面积小的都忽略
    #     if(area < 100):
    #         continue
    #     #连接轮廓点并画出来
    #     cv2.polylines(imagel, [cnt], True, (0, 255, 0), 2)
    #     #计算周长
    #     peri = cv2.arcLength(cnt, True)
    #     #求出轮廓中心点
    #     approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    #     #画出中心点
    #     cv2.circle(imagel, (approx[0][0][0], approx[0][0][1]), 5, (0, 0, 255), -1)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:
            continue
        
        rect = cv2.minAreaRect(cnt)  # 求最小外接矩形
        box = cv2.boxPoints(rect)  # 返回矩形的四个顶点
        box = np.int0(box)  # 转换为整型
        
        cv2.drawContours(imagel, [box], 0, (255, 0, 0), 2)  # 画出矩形
        
        M = cv2.moments(cnt)  # 计算矩值，方便得到矩形中心点


        if M["m00"] != 0:
            
            regX = clX
            regY = clY
            clX = int(M["m10"] / M["m00"])
            clY = int(M["m01"] / M["m00"])

            cv2.circle(imagel, (clX, clY), 5, (0, 0, 255), -1)  # 画出矩形中心点
            if clX != regX or clY != regY:
                print(clX, clY)

    cv2.imshow('imagel', imagel)
    cv2.imshow('imagel_dilated', imagel_dilated)

    time.sleep(0.1)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()