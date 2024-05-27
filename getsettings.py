import cv2

cap = cv2.VideoCapture(1)
num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
high = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

print("Number of frames: ", num)
print("Frame height: ", high)
print("Frame width: ", width)