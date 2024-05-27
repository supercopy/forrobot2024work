from orangeblock import fruit_center

if __name__ == "__main__":
    stereo_vision = fruit_center(2, 1)
    fruit_centerl, fruit_centerr = stereo_vision.detect_objects() #得到目标点坐标
    print(fruit_centerl, fruit_centerr)

    #depth = calculate_depth(disparity, clX, clY, crX, crY, camera_parameters)