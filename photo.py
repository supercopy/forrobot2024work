import cv2
import time
# 定义拍照函数类
class Photo:
    def __init__(self, camera_idx = 0, num = 1): # camera_idx的作用是选择摄像头。如果为0则使用内置摄像头，比如笔记本的摄像头，用1或其他的就是切换摄像头。
        self.cap = cv2.VideoCapture(camera_idx)
        self.ret, self.frame = self.cap.read() 
    def snapShotCt(self, num):
        for i in range(num):
            self.ret,self.frame = self.cap.read() # 下一个帧图片
            cv2.imwrite('pics\photo' + str(i) + ".jpg", self.frame) # 写入图片
            # 输出拍摄的是第几张照片
            print("已拍摄第", i, "张照片") 
            cv2.imshow("SnapShot", self.frame)# 显示图片
            time.sleep(0.01) # 休眠一秒 可通过这个设置拍摄间隔，类似帧.
            if self.ret == False: # 如果拍摄失败，则退出循环
                break
                
if __name__ == '__main__':
    #创建photo类对象
    cap = cv2.VideoCapture(0)
    tar_num = 0
    current_pooto_num = int(input("请输入当前照片数量："))  
    while (True):
        num  = int(input("请输入要拍摄的照片数量："))         
        tar_num = current_pooto_num + num #要拍摄的照片数量     
        current_pooto_num = tar_num - num #当前拍摄的照片数量   
        for i in range(current_pooto_num ,tar_num):
            ret,frame = cap.read() # 下一个帧图片
            cv2.imwrite('C:/Users/32837/Desktop/MyYoloTrain/5.photo/images/' + str(i + 1) + ".jpg", frame) # 写入图片
            # 输出拍摄的是第几张照片
            print("已拍摄第", i + 1, "张照片") 
            cv2.imshow("SnapShot", frame)# 显示图片  
            time.sleep(0.1) # 休眠一秒 可通过这个设置拍摄间隔，类似帧。
            if ret == False: # 如果拍摄失败，则退出循环
                break     
        current_pooto_num = tar_num 
        if cv2.waitKey(1) & 0xff == ord('e'):  # 按e键退出，可以改成任意键
            cap.release()
            cv2.destroyAllWindows()
            break
            
# import time
# import cv2
# import numpy as np
 
# def takephoto():
#     cap = cv2.VideoCapture(1)
#     index = 0
#     ret, frame = cap.read()  # ret为布尔型，表示有咩有读取到图片，frame表示截取到一帧的图片
#     while ret:
#         for index in range(3):
#             resize = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_NEAREST)
#             cv2.imwrite(str(index) + '.jpg', resize)
#             img = cv2.imread(filename=str(index) + '.jpg')
#             cv2.imshow("photo", img)
#             cv2.waitKey(3000)
#             cv2.destroyAllWindows()#多这一条语句为防止图片连续，效果为重新弹出窗口
#             time.sleep(2)
#             ret, frame = cap.read()
#             index += 1
#         ret = False
 
#     cap.release()
#     cv2.destroyAllWindows()
#     return 0
 
 
# if __name__ == '__main__':
# #实时预览
#     cap = cv2.VideoCapture(0)  # 0代表树莓派上自带的摄像头，1代表USB摄像头
 
#     # 一下cap.set(),可以注释掉#
#     # cap.set(3, 320)  # 摄像头采集图像的宽度320
#     # cap.set(4, 240)  # 摄像头采集图像的高度240
#     # cap.set(5, 90)  # 摄像头采集图像的帧率fps为90
 
#     # 查看采集图像的参数
#     # print(cap.get(3))
#     # print(cap.get(4))
#     # print(cap.get(5))#只有15帧
 
#     while (True):
#         ret, color_frame = cap.read()
#         img1 = cv2.flip(color_frame, 1)  # 翻转图像，0垂直翻转，1水平翻转，-1水平垂直翻转
#         cv2.imshow('color_frame', img1)  # 展示每一帧
#         if cv2.waitKey(1) & 0xff == ord('e'):  # 按e键退出，可以改成任意键
#             break
#     cap.release()
#     cv2.destroyAllWindows()
# #开始拍照
#     print('Begin to take pictures..........')
#     takephoto()
#     print('Finished')