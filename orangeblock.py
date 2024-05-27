import cv2
import numpy as np

class fruit_center:
    def __init__(self, left_image, right_image):

        # 初始化参数
        self.left_image = left_image
        self.right_image = right_image
        self.kernel = np.ones((5,5), np.uint8)
        self.kernel_open = np.ones((10,10), np.uint8)
        self.hsv_low = np.array([13, 105, 83])
        self.hsv_high = np.array([31, 255, 255])
        self.clX,self.clY = 0, 0
        self.crX,self.crY = 0, 0

    def get_contours(self, image): #图像处理并返回轮廓点
        image = cv2.GaussianBlur(image, (5, 5), 0)
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image_dst = cv2.inRange(image_hsv, self.hsv_low, self.hsv_high)
        image_open = cv2.morphologyEx(image_dst, cv2.MORPH_OPEN, self.kernel_open)
        image_dilated = cv2.dilate(image_open, self.kernel, iterations=1)
        contours, _ = cv2.findContours(image_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours, image_dilated
    
    #求最小外接矩形并返回顶点，使用该函数在遍历轮廓内
    def get_rectangle(self, cnt):
        rect = cv2.minAreaRect(cnt) #求最小外接矩形
        box = cv2.boxPoints(rect) #返回矩形的四个顶点
        box = np.int0(box) #转换为整型
        return box

    def get_center(self, moments): #通过矩值计算中心点
        centerX = int(moments["m10"] / moments["m00"])
        centerY = int(moments["m01"] / moments["m00"])
        return centerX, centerY

    def detect_objects(self):
        centerl_points = None
        centerr_points = None
        imagel = self.left_image
        imager = self.right_image

        contoursl, imagel_dilated = self.get_contours(imagel)
        contoursr, imager_dilated = self.get_contours(imager)

        for cntl in contoursl:
            areal = cv2.contourArea(cntl)
            if areal < 100:
                continue
            boxl = self.get_rectangle(cntl)  # 求最小外接矩形
            cv2.drawContours(imagel, [boxl], 0, (255, 0, 0), 2)  # 画出矩形
            Ml = cv2.moments(cntl)  # 计算矩值，方便得到矩形中心点

            if Ml["m00"] != 0:
                reglX = self.clX
                reglY = self.clY
                self.clX, self.clY = self.get_center(Ml)  # 得到矩形中心点
                cv2.circle(imagel, (self.clX, self.clY), 5, (0, 0, 255), -1)  # 画出矩形中心点
                if self.clX != reglX or self.clY != reglY:
                    centerl_points = (self.clX, self.clY)  # 得到左相机目标点坐标
                    print("左相机目标点坐标：",centerl_points)

        for cntr in contoursr:
            arear = cv2.contourArea(cntr)
            if arear < 100:
                continue
            boxr = self.get_rectangle(cntr)  # 求最小外接矩形
            cv2.drawContours(imager, [boxr], 0, (255, 0, 0), 2)  # 画出矩形
            Mr = cv2.moments(cntr)  # 计算矩值，方便得到矩形中心点
            
            if Mr["m00"] != 0:
                regrX = self.crX
                regrY = self.crY
                self.crX, self.crY = self.get_center(Mr)  # 得到矩形中心点
                cv2.circle(imager, (self.crX, self.crY), 5, (0, 0, 255), -1)  # 画出矩形中心点
                if self.crX != regrX or self.crY != regrY:
                        centerr_points = (self.crX, self.crY)  # 得到右相机目标点坐标
                        print("右相机目标点坐标：",centerr_points)

        cv2.imshow('imagel', imagel)
        cv2.imshow('imager', imager)
        cv2.imshow('imagel_dilated', imagel_dilated)
        cv2.imshow('imager_dilated', imager_dilated)

        return self.clX, self.clY, self.crX, self.crY