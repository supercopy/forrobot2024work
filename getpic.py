import cv2
import os

# 创建素材子文件夹
if not os.path.exists('pics'):
    os.makedirs('pics')

# 打开摄像头
cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cap1 = cv2.VideoCapture(1)
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

i = 1
j = 1

while True:
    ret1, framel = cap.read(1)
    ret2, framer = cap1.read(1)
    
    # 角点检测
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # corners = cv2.goodFeaturesToTrack(gray, 100, 0.3, 10)
    # rframe = frame[200:390, 330:600]
    
    
    # 检测按键
    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):
        filenamel = f'pics/left/cornerl_{i}.png'
        filenamer = f'pics/right/cornerr_{i}.png'
        # filenamel = f'pics/cornerl_{i}.png'
        # filenamer = f'pics/cornerr_{i}.png'
        cv2.imwrite(filenamel, framel)
        cv2.imwrite(filenamer, framer)
        print(f'Frame captured and saved as {filenamel}!')
        print(f'Frame captured and saved as {filenamer}!')
        i += 1
    # if key == ord('r'):
    #     filenamer = f'pics/cornerr_{j}.png'
    #     cv2.imwrite(filenamer, framer)
    #     print(f'Frame captured and saved as {filenamer}!')
    #     j += 1
    
    if key == 27:  # 按下ESC键退出
        break
    
    # if corners is not None:
    #     corners = corners.reshape(-1, 2)
    #     for corner in corners:
    #         x, y = corner
    #         cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

    cv2.imshow('Framel', framel)
    cv2.imshow('Framer', framer)
    # cv2.imshow('Cropped Frame', rframe)

cap.release()
cv2.destroyAllWindows()