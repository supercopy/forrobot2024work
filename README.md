# forrobot2024work_D组
小组共享  
注意！！！：  
使用代码时请人工识别左右相机对应编号，根据实际情况修改代码  
因为使用识别方法为传统阈值检测，对背景要求比较苛刻，请在每次使用时查看相机实时捕获的图像在二值化后是否可用  
所用双目相机拍照标准尺寸为1280*720，可根据需要设置相机参数，相机需要手动对焦，其余参数可使用电脑自带相机app拍照后查看照片参数获得  
  
文件说明：  
getpic.py 同时获取左右目图片存储到目标文件夹  
getmat.py 读取来自matlab的mat数据文件（个人评价：看不明白，人工组上大分）  
get_threshold.py 阈值调试工具，请注意：改变光照之后的阈值可能需要重新调试  
change_name.py 快速修改文件名字  
orange.py 返回橘子轮廓中心坐标  
getsettings.py 得到当前相机宽高参数  
orangeblock.py 可调用的orange类  
mysgbm-video.py 调用orangeblock.py返回的二维坐标进行归一化变化，根据左右相机深度图得到目标点深度  
使用transform.m 求出变换矩阵，可添加更多点提升精确度，estimate_transform_matrix.m为使用最小二乘法估计变换矩阵T_AB的函数，需要一同下载使用  
使用reg.py验证变换矩阵  
  
需要UI:  
请查看mysgbm-video.py 
使用槽函数实现输入相机序列号，方便左右相机的选择；  
显示目标在左右相机的像素坐标，即(clX，clY),(crX, crY)；  
显示计算得到的视差；  
显示目标在左右相机深度图的坐标，显示求出的深度；  
  
建议在类中预留:  
显示目标数目，排序并显示三维坐标，目前采摘到第几个  
