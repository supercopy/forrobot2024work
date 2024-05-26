import cv2
import numpy as np
import time

kernel = np.ones((5,5),np.uint8)
kernel_open = np.ones((10,10),np.uint8)

cap = cv2.VideoCapture(1)#左相机
cap.set(3, 640) #设置分辨率宽
cap.set(4, 480) #设置分辨率高
#[ 13 105  83] [ 31 255 255]

hsv_low = np.array([13, 105, 83])
hsv_high = np.array([31, 255, 255])

cX = 1
cY = 1

while(True):
    ret, frame = cap.read()
    image = frame.copy()

    #高斯模糊
    image = cv2.GaussianBlur(image, (5,5), 0)
    #转换为HSV
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #二值化
    image_dst = dst = cv2.inRange(image_hsv, hsv_low, hsv_high)
    #开运算
    image_open = cv2.morphologyEx(image_dst, cv2.MORPH_OPEN, kernel_open)
    #膨胀操作
    image_dilated = cv2.dilate(image_open, kernel, iterations=1)
    #查找轮廓
    contours, hierarchy = cv2.findContours(image_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # for cnt in contours:
    #     #计算轮廓的面积
    #     area = cv2.contourArea(cnt)
    #     #面积小的都忽略
    #     if(area < 100):
    #         continue
    #     #连接轮廓点并画出来
    #     cv2.polylines(image, [cnt], True, (0, 255, 0), 2)
    #     #计算周长
    #     peri = cv2.arcLength(cnt, True)
    #     #求出轮廓中心点
    #     approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    #     #画出中心点
    #     cv2.circle(image, (approx[0][0][0], approx[0][0][1]), 5, (0, 0, 255), -1)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:
            continue
        
        rect = cv2.minAreaRect(cnt)  # 求最小外接矩形
        box = cv2.boxPoints(rect)  # 返回矩形的四个顶点
        box = np.int0(box)  # 转换为整型
        
        cv2.drawContours(image, [box], 0, (255, 0, 0), 2)  # 画出矩形
        
        M = cv2.moments(cnt)  # 计算矩值，方便得到矩形中心点


        if M["m00"] != 0:
            
            regX = cX
            regY = cY
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)  # 画出矩形中心点
            if cX != regX or cY != regY:
                print(cX, cY)

    cv2.imshow('image', image)
    cv2.imshow('image_dst', image_dst)
    cv2.imshow('image_open', image_open)
    cv2.imshow('image_dilated', image_dilated)

    time.sleep(0.1)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()