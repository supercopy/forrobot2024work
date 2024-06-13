import cv2
import numpy as np # type: ignore
import time
from tqdm import tqdm
from ultralytics import YOLO
import matplotlib.pyplot as plt
import torch
# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device:', device)

# 加载模型
model = YOLO('C:/Users/13472/Desktop/robot/MyYoloTrain/model_train_test/runs/detect/train/weights/best.pt')
model.cuda() # 将模型移动到 GPU 上

# 框（rectangle）可视化配置
bbox_color = (150, 0, 0)             # 框的 BGR 颜色
bbox_thickness = 2                   # 框的线宽

# 框中心点
def get_center(x1, y1, x2, y2):
    return (x1 + x2) / 2, (y1 + y2) / 2

# 框类别文字
bbox_labelstr = {
    'font_size':2,         # 字体大小
    'font_thickness':4,   # 字体粗细
    'offset_x':0,          # X 方向，文字偏移距离，向右为正
    'offset_y':-20,       # Y 方向，文字偏移距离，向下为正
}
# # 关键点 BGR 配色
kpt_color_map = {
    0:{'name':'Nose', 'color':[0, 0, 255], 'radius':6},                # 鼻尖
    1:{'name':'Right Eye', 'color':[255, 0, 0], 'radius':6},           # 右边眼睛
    2:{'name':'Left Eye', 'color':[255, 0, 0], 'radius':6},            # 左边眼睛
    3:{'name':'Right Ear', 'color':[0, 255, 0], 'radius':6},           # 右边耳朵
    4:{'name':'Left Ear', 'color':[0, 255, 0], 'radius':6},            # 左边耳朵
    5:{'name':'Right Shoulder', 'color':[193, 182, 255], 'radius':6},  # 右边肩膀
    6:{'name':'Left Shoulder', 'color':[193, 182, 255], 'radius':6},   # 左边肩膀
    7:{'name':'Right Elbow', 'color':[16, 144, 247], 'radius':6},      # 右侧胳膊肘
    8:{'name':'Left Elbow', 'color':[16, 144, 247], 'radius':6},       # 左侧胳膊肘
    9:{'name':'Right Wrist', 'color':[1, 240, 255], 'radius':6},       # 右侧手腕
    10:{'name':'Left Wrist', 'color':[1, 240, 255], 'radius':6},       # 左侧手腕
    11:{'name':'Right Hip', 'color':[140, 47, 240], 'radius':6},       # 右侧胯
    12:{'name':'Left Hip', 'color':[140, 47, 240], 'radius':6},        # 左侧胯
    13:{'name':'Right Knee', 'color':[223, 155, 60], 'radius':6},      # 右侧膝盖
    14:{'name':'Left Knee', 'color':[223, 155, 60], 'radius':6},       # 左侧膝盖
    15:{'name':'Right Ankle', 'color':[139, 0, 0], 'radius':6},        # 右侧脚踝
    16:{'name':'Left Ankle', 'color':[139, 0, 0], 'radius':6},         # 左侧脚踝
}
#处理单帧图像可视化配置
def process_frame(img_bgr):
    
    '''
    输入摄像头画面 bgr-array，输出图像 bgr-array
    '''
    # 记录该帧开始处理的时间
    start_time = time.time()
    
    results = model(img_bgr, verbose=False,conf = 0.8) # verbose设置为False，不单独打印每一帧预测结果    
    # 预测框的个数
    num_bbox = len(results[0].boxes.cls)
    
    # 预测框的 xyxy 坐标
    bboxes_xyxy = results[0].boxes.xyxy.cpu().numpy().astype('uint32') 
    
    # 关键点的 xy 坐标
    # bboxes_keypoints = results[0].keypoints.data.cpu().numpy().astype('uint32')
    bbox_xyxy = [0,0,0,0,0]

    for idx in range(num_bbox): # 遍历每个框

        # 获取该框坐标
        bbox_xyxy = bboxes_xyxy[idx] 

        # 获取框的预测类别（对于关键点检测，只有一个类别）
        bbox_label = results[0].names[0]

        # 画框
        img_bgr = cv2.rectangle(img_bgr, (bbox_xyxy[0], bbox_xyxy[1]), (bbox_xyxy[2], bbox_xyxy[3]), bbox_color, bbox_thickness)

        # 写框类别文字：图片，文字字符串，文字左上角坐标，字体，字体大小，颜色，字体粗细
        img_bgr = cv2.putText(img_bgr, bbox_label, (bbox_xyxy[0]+bbox_labelstr['offset_x'], bbox_xyxy[1]+bbox_labelstr['offset_y']), cv2.FONT_HERSHEY_SIMPLEX, bbox_labelstr['font_size'], bbox_color, bbox_labelstr['font_thickness'])
        for result in results:
            boxes = result.boxes
            for box in boxes:
                confidence = box.conf.item()
                img_bgr = cv2.putText(img_bgr,str(f"{confidence:.2f}") , (bbox_xyxy[0]+2*bbox_labelstr['offset_x'], bbox_xyxy[1]+2*bbox_labelstr['offset_y']), cv2.FONT_HERSHEY_SIMPLEX, bbox_labelstr['font_size'], bbox_color, bbox_labelstr['font_thickness'])

    # 记录该帧处理完毕的时间
    end_time = time.time()
    # 计算每秒处理图像帧数FPS
    FPS = 1/(end_time - start_time)

    # 在画面上写字：图片，字符串，左上角坐标，字体，字体大小，颜色，字体粗细
    FPS_string = 'FPS  '+str(int(FPS)) # 写在画面上的字符串
    img_bgr = cv2.putText(img_bgr, FPS_string, (25, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 0, 255), 2)
    return img_bgr,bbox_xyxy
# 调用摄像头逐帧实时处理模板
# 不需修改任何代码，只需修改process_frame函数即可

# 导入opencv-python
import cv2
import time

# 获取摄像头，传入0表示获取系统默认摄像头
cap = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(2)

# 打开cap
cap.open(1)
cap2.open(2)

# 无限循环，直到break被触发
while cap.isOpened()&cap2.isOpened():
    
    # 获取画面
    success, frame = cap.read()
    success2, frame2 = cap2.read()
    
    if not success:# |  success2: # 如果获取画面不成功，则退出
        print('获取画面不成功，退出')
        break
    
    ## 逐帧处理
    frame,bbox_xyxy = process_frame(frame)  
    frame2,bbox_xyxy2 = process_frame(frame2)
    # 计算中心点
    CenterOrange_x1,CenterOrange_y1 = get_center(bbox_xyxy[0],bbox_xyxy[1],bbox_xyxy[2],bbox_xyxy[3])
    CenterOrange_x2,CenterOrange_y2 = get_center(bbox_xyxy2[0],bbox_xyxy2[1],bbox_xyxy2[2],bbox_xyxy2[3])

    
    
    ##################测试区############################
    #测试区
    # for i in range(4):
    #      print(bbox_xyxy[i])
    #      print(bbox_xyxy2[i])
    ##################测试区############################
    # 展示处理后的三通道图像
    imgStackH = np.hstack((frame, frame2))

    cv2.imshow('my_window',imgStackH)
    
    key_pressed = cv2.waitKey(60) # 每隔多少毫秒毫秒，获取键盘哪个键被按下
    # print('键盘上被按下的键：', key_pressed)

    if key_pressed in [ord('q'),27]: # 按键盘上的q或esc退出（在英文输入法下）
        break
    
# 关闭摄像头
cap.release()

# 关闭图像窗口
cv2.destroyAllWindows()