
import rtde_control
import rtde_io
import time
import rtde_receive

whoes = "192.168.1.11"

#位姿
pose1 = [-0.25020898611400244, -0.4881749740093524, 0.0033911165375497543+0.05, 1.0454271042245586, -0.7643593437133561, -2.411152011171747] #左一重新定参
pose2 = [-0.24826298305699612, -0.5988960759882378, 0.03858996361983327, 1.0452817703443196, -0.7644291461957899, -2.410808940486624]
pose3 = [-0.363311293655781, -0.5937462576694004, 0.0427015804086833, 1.0454591285654575, -0.7647099511598795, -2.4110825367188773]
pose4 = [-0.36593478450893047, -0.4835069242463028, 0.04506044850732732, 1.0453466985348745, -0.7645599768318118, -2.411050059172221]
pose5 = [0.3214552588517872, -0.4596778114833889, 0.0009474690967989086+0.05, 1.0896884484104383, -0.5283030328096147, -2.1280760780320063]#右一重新定参
pose6 = [0.3580649877104597, -0.5612679469788149, 0.049686952694683595, 1.0885939539820095, -0.5406056483870899, -2.144695062358497]
pose7 = [0.25290133621224076, -0.6007140896253769, 0.05259986325145083, 1.0885767310078918, -0.5407492631775866, -2.1446372598745307]
pose8 = [0.21345740748656133, -0.4998935534560503, 0.0560583707453815, 1.0885165395882697, -0.5407433380734024, -2.1446446411198323]

def movedown(arr1): #移动到对应位姿并下降，服务于夹持
    rtde_c = rtde_control.RTDEControlInterface(whoes) 
    rtde_c.moveL(arr1, 0.8, 0.6)
    time.sleep(0.1)
    reg = arr1
    reg[2] = reg[2] - 0.045
    rtde_c.moveL(reg, 0.8, 0.6)
    time.sleep(0.1)

def moveup(num): #在当前位姿上升0.045m，避免碰撞
    rtde_r = rtde_receive.RTDEReceiveInterface(whoes)
    rtde_c = rtde_control.RTDEControlInterface(whoes)
    actual = rtde_r.getActualTCPPose()
    actual[2] = actual[2] + num
    rtde_c.moveL(actual, 0.8, 0.6)
    time.sleep(0.1)

def moveto(arr): #移动到arr
    rtde_c = rtde_control.RTDEControlInterface(whoes) 
    rtde_c.moveL(arr, 0.8, 0.6)
    time.sleep(0.1)

def clamp(): #夹取
    rtde_io8 = rtde_io.RTDEIOInterface(whoes)
    rtde_io8.setStandardDigitalOut(0, True)
    time.sleep(0.1)

def release(): #释放
    rtde_io8 = rtde_io.RTDEIOInterface(whoes)
    rtde_io8.setStandardDigitalOut(0, False)
    time.sleep(0.1)

# #得到当前姿态
# rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.10")
# actual_q = rtde_r.getActualTCPPose()
# print(actual_q)
# time.sleep(0.3)

def move(arr1, arr2): #移动到arr1并夹取并移动到arr2释放
    movedown(arr1)
    clamp()
    moveup(0.045)
    movedown(arr2)
    release()
    moveup(0.1)

def dataprocess(num): #数据处理,0正序，1倒序，其余提示输入错误
    hundred = (int)(num / 100)
    ten = (int)((num - 100*hundred)/10)
    single = num % 10
    if hundred == 0:
        datas = [hundred, ten, single]
    elif hundred == 1:
        datas = [hundred, single, ten]
    else:
        datas = [hundred, 0, 0]
    return datas

def sift(data): #筛选对应位置
    if data == 1:
        return pose1
    if data == 2:
        return pose2
    if data == 3:
        return pose3
    if data == 4:
        return pose4
    if data == 5:
        return pose5
    if data == 6:
        return pose6
    if data == 7:
        return pose7
    if data == 8:
        return pose8
    
def final(datas): #最终函数
    if datas[0] != 0 and datas[0] != 1:
        print("INPUT ERRORS") #后续修改
    else:
        movef = sift(datas[1])
        moveb = sift(datas[2])
        move(movef, moveb)

if __name__ == '__main__': 
    while True:
        try:
            sum = int(input("请输入数字:"))
            break  # 如果用户输入的是整数，则跳出循环
        except ValueError:
            print("请输入一个有效的整数！")
    time.sleep(1)
    dataing = dataprocess(sum)
    final(dataing)


