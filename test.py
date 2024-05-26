import cv2
import tkinter as tk
from tkinter import Scale

# 创建窗口
root = tk.Tk()
root.title("Camera Parameters Adjuster")

# 打开摄像头
cap = cv2.VideoCapture(0)

# 定义参数范围
parameter_ranges = {
    cv2.CAP_PROP_BRIGHTNESS: (0, 255),
    cv2.CAP_PROP_CONTRAST: (0, 255),
    cv2.CAP_PROP_SATURATION: (0, 255),
    cv2.CAP_PROP_HUE: (0, 179)
}

def update_parameter(value, parameter):
    cap.set(parameter, value)

# 创建滑块控件
for parameter, (min_val, max_val) in parameter_ranges.items():
    scale = Scale(root, label=str(parameter), from_=min_val, to=max_val, orient=tk.HORIZONTAL,
                  command=lambda value, param=parameter: update_parameter(value, param))
    scale.pack()

def update_frame():
    ret, frame = cap.read()
    cv2.imshow('Frame', frame)
    root.after(10, update_frame)

update_frame()

root.mainloop()

cap.release()
cv2.destroyAllWindows()

import cv2

# 创建摄像头对象
cap = cv2.VideoCapture(0)

# 恢复摄像头默认参数
cap.set(cv2.CAP_PROP_BRIGHTNESS, 0)
cap.set(cv2.CAP_PROP_CONTRAST, 1)
cap.set(cv2.CAP_PROP_SATURATION, 1)
cap.set(cv2.CAP_PROP_HUE, 0)
cap.set(cv2.CAP_PROP_GAIN, 0)
cap.set(cv2.CAP_PROP_EXPOSURE, -1)
cap.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, -1)
cap.set(cv2.CAP_PROP_WHITE_BALANCE_RED_V, -1)
cap.set(cv2.CAP_PROP_BACKLIGHT, 0)
cap.set(cv2.CAP_PROP_SHARPNESS, 0)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
cap.set(cv2.CAP_PROP_AUTO_WB, 1)

# 使用摄像头进行操作
while True:
    ret, frame = cap.read()
    
    # 在这里进行摄像头操作
    # ...
    
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头对象
cap.release()
cv2.destroyAllWindows()