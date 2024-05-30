import rtde_control
import rtde_io
import time
import rtde_receive

whoes = "192.168.1.11"

def gettcp(who):
    rtde_r = rtde_receive.RTDEReceiveInterface(who)
    actual = rtde_r.getActualTCPPose()
    actual_xyz = [actual.x, actual.y, actual.z]
    actual_rpw = [actual.r, actual.p, actual.w]
    return actual_xyz, actual_rpw

actual_xyz, actual_rpw = gettcp(whoes)
print(actual_xyz)
print(actual_rpw)