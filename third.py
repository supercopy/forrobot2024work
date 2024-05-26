import rtde_control
import rtde_io
import time
import rtde_receive
import numpy as np

whoes = "192.168.1.10"

rtde_r = rtde_receive.RTDEReceiveInterface(whoes)
rtde_c = rtde_control.RTDEControlInterface(whoes)
# rtde_c.moveL(arr1, 0.8, 0.6)

def addd(arr1, arr2):
    result = []
    for num1, num2 in zip(arr1, arr2):
        result.append(num1 + num2)
    return result

#参数-num要写的数字   ，width为宽度（mm）
def To(num, width):
    if num == 4:
        orgPos=rtde_r.getActualTCPPose() # orgPos 为左下角点+5mm
        z = addd(orgPos , [6*width/8000,13*width/8000,0,0,0,0])
        a = addd(orgPos , [6*width/8000,13*width/8000,-5/1000,0,0,0])
        b = addd(orgPos , [0*width/8000,6*width/8000,-5/1000,0,0,0])
        c = addd(orgPos , [7*width/8000,6*width/8000,-5/1000,0,0,0])
        d = addd(orgPos , [7*width/8000,6*width/8000,0/1000,0,0,0])
        e = addd(orgPos , [6*width/8000,13*width/8000,0,0,0,0])
        f = addd(orgPos , [6*width/8000,13*width/8000,-5/1000,0,0,0])
        g = addd(orgPos , [6*width/8000,2*width/8000,-5/1000,0,0,0])
        h = addd(orgPos , [6*width/8000,2*width/8000,0/1000,0,0,0])
        i = addd(orgPos , [8*width/8000,0*width/8000,0/1000,0,0,0])
        rtde_c.moveL(z, 0.8, 0.6) #z
        rtde_c.moveL(a, 0.8, 0.6) #a
        rtde_c.moveL(b, 0.8, 0.6) #b
        rtde_c.moveL(c, 0.8, 0.6) #c
        rtde_c.moveL(d, 0.8, 0.6) #d
        rtde_c.moveL(e, 0.8, 0.6) #e
        rtde_c.moveL(f, 0.8, 0.6) #f
        rtde_c.moveL(g, 0.8, 0.6) #g
        rtde_c.moveL(h, 0.8, 0.6) #h
        rtde_c.moveL(i, 0.8, 0.6) #i
    elif num == 2:
        orgPos=rtde_r.getActualTCPPose() # orgPos 为左下角点+5mm
        

# if __name__ == '__main__': 
#     while True:
#         try:
#             sum = int(input("请输入数字:"))
#             break  # 如果用户输入的是整数，则跳出循环
#         except ValueError:
#             print("请输入一个有效的整数！")
#     time.sleep(1)

# orgPos=rtde_r.getActualTCPPose() # orgPos 为左下角点+5mm
# z = addd(orgPos , [6*50/8000,13*50/8000,0,0,0,0])
# print(orgPos)
# print(z)

To(50)
