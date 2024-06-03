import rtde_control
import rtde_io
import time
import rtde_receive

whoes = "192.168.1.11"

def gettcp(who):
    rtde_r = rtde_receive.RTDEReceiveInterface(who)
    actual = rtde_r.getActualTCPPose()
    actual_xyz = [actual[0], actual[1], actual[2]]
    actual_rpw = [actual[3], actual[4], actual[5]]
    return actual_xyz, actual_rpw

actual_xyz, actual_rpw = gettcp(whoes)
print(actual_xyz)
print(actual_rpw)

