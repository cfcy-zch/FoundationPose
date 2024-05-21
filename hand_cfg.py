import numpy as np
import math

# 机械手参数
unit_length = 1000
module1_config = {
    "p1": np.array([5.355 / unit_length, -6. / unit_length, 17.185 / unit_length]),
    "p2": np.array([5.355 / unit_length, 6. / unit_length, 17.185 / unit_length]),
    "p3": np.array([14.050 / unit_length, -6. / unit_length, 0]),
    "p4": np.array([14.050 / unit_length, 6. / unit_length, 0]),
    "l1": 22.9 / unit_length,
    "l2": 22.9 / unit_length,
    "z0": 4. / unit_length,
    "bevel": 256 * 64 * 2,
    # "screw": 2 * math.pi / 0.001,  
    "h": 0.001,
    "screw": 1 * 256 * 4 / 0.001, 
    "A": 9.5 / unit_length,
    "B": 32.35 / unit_length,
    "C": 29.867 / unit_length,
    "D": 7.3 / unit_length,
    "degree1_init": math.radians(61.4845),
    "degree2_init": math.radians(118.1266),
}
module2_config = {
    "p1": np.array([9.932 / unit_length, 0. / unit_length, 13.797 / unit_length]),
    "p2": np.array([0. / unit_length,  1. / unit_length, 0. / unit_length]),
    "p3": np.array([13.050 / unit_length, 0. / unit_length, 0]),
    "l1": 5.6 / unit_length,
    "l2": 18.4 / unit_length,
    "z0": 4. / unit_length,
    "bevel": 128 * 256 * 2,
    "screw": 4 * 256 * 4 / 0.001,   
    "h": 0.001,
    "A": 4.5 / unit_length,
    "B": 26.5 / unit_length,
    "C": 25.  / unit_length,
    "D": 6. / unit_length,
    "degree1_init": math.radians(45.0619),
    "degree2_init": math.radians(124.5317),
    # mcp-pip
    "Am": 7. / unit_length,
    "Bm": 46. / unit_length,
    "Cm": 45. / unit_length,
    "Dm": 6. / unit_length,
    "degree1m_init": math.radians(66.3557),
    "degree2m_init": math.radians(111.6954),
    
}
module3_config = {
    "p1": np.array([9.932 / unit_length, 0. / unit_length, 13.797 / unit_length]),
    "p2": np.array([0. / unit_length,  1. / unit_length, 0. / unit_length]),
    "p3": np.array([13.050 / unit_length, 0. / unit_length, 0]),
    "l1": 5.6 / unit_length,
    "l2": 18.4 / unit_length,
    "z0": 4. / unit_length,
    "bevel": 128 * 256 * 2,
    "screw": 4 * 256 * 4 / 0.001, 
    "h": 0.001,
    "A": 4.5 / unit_length,
    "B": 26.5 / unit_length,
    "C": 25. / unit_length,
    "D": 6. / unit_length,
    "degree1_init": math.radians(45.0619),
    "degree2_init": math.radians(124.5317),
    # mcp-pip
    "Am": 7. / unit_length,
    "Bm": 36.85 / unit_length,
    "Cm": 36. / unit_length,
    "Dm": 6. / unit_length,
    "degree1m_init": math.radians(67.63534),
    "degree2m_init": math.radians(103.46481),
}
module4_config = {
    # "screw": 1 * 256 * 4 / 0.001, 
    "h": 0.001,
    "A": 7. / unit_length,
    "B": 37.61 / unit_length,
    "C": 36. / unit_length,
    "D": 5.5 / unit_length,
    "degree1_init": math.radians(70.96005765), # 上一版71.98912
    "degree2_init": math.radians(111.50535364),
    
    
    "A1": 7.10828714 / unit_length,#横向距离
    "B1": 8.5 / unit_length,#纵向距离
    "degree1_init1": math.radians(50.09530324),#水平夹角
    "A2": 7.20179328 / unit_length,
    "B2": 9 / unit_length,
    "degree1_init2": math.radians(51.3332312),
}
module_config = {
    "thumb": module4_config,
    "index": module1_config,
    "middle": module1_config,
    "fourth": module2_config,
    "little": module3_config,
}
finger_name = ["index","middle","fourth","little","thumb"]
motor_mapping = {
        "index" : [9,8,7],
        "middle": [5,10,6],
        "fourth": [2,4],
        "little": [1,3],
        "thumb" : [0,11,12],
        "all"   : [9,8,7,5,10,6,2,4,1,3,0,11,12],
}
motor_direction = {
        "index" : [1,1,-1],
        "middle": [1,1,-1],
        "fourth": [1,1],
        "little": [1,1],
        "thumb" : [-1,-1,1],
        "all"   : [-1, 1, 1, 1, 1, 1,
                   -1, -1, 1, 1, 1, -1, 1],
}
ratio = {
        "index" : [12 * 4 * 4,
                    12 * 4 * 4,
                    12 * 256 * 2 * 4],
        "middle": [12 * 4 * 4,
                    12 * 4 * 4,
                    12 * 256 * 2 * 4],
        "fourth": [12 * 256 * 2 * 4,
                    12 * 4 * 4,],
        "little": [12 * 256 * 2 * 4,
                    12 * 4 * 4],
        "thumb" : [16 * 90.21 * 2 * 4,
                    12 * 16 * 4,
                    12 * 16 * 4],
        "all"   : [12 * 4 * 4,
                    12 * 4 * 4,
                    12 * 256 * 2 * 4,
                    12 * 4 * 4,
                    12 * 4 * 4,
                    12 * 256 * 2 * 4,
                    12 * 256 * 2 * 4,
                    12 * 4 * 4,
                    12 * 256 * 2 * 4,
                    12 * 4 * 4,
                    16 * 90.21 * 4,
                    12 * 16 * 4,
                    12 * 16 * 4],
}
# ratio = {
#         "index" : [12 * 4 * 4,
#                     12 * 4 * 4,
#                     12 * 256 * 2 * 4],
#         "middle": [12 * 4 * 4,
#                     12 * 4 * 4,
#                     12 * 256 * 2 * 4],
#         "fourth": [12 * 256 * 2 * 4,
#                     12 * 4 * 4,],
#         "little": [12 * 256 * 2 * 4,
#                     12 * 4 * 4],
#         "thumb" : [16 * 90.21 * 8,
#                     12 * 16 * 100,
#                     12 * 16 * 100],
#         "all"   : [12 * 4 * 4,
#                     12 * 4 * 4,
#                     12 * 256 * 2 * 4,
#                     12 * 4 * 4,
#                     12 * 4 * 4,
#                     12 * 256 * 2 * 4,
#                     12 * 256 * 2 * 4,
#                     12 * 4 * 4,
#                     12 * 256 * 2 * 4,
#                     12 * 4 * 4,
#                     16 * 90.21 * 8,
#                     12 * 16 * 100,
#                     12 * 16 * 100],
# }

MOTOR_NUM = 13
class FingerCommand:
    def __init__(self):
        self.zero()

    def zero(self):    
        self.qDes = np.zeros(MOTOR_NUM)
        self.qdDes = np.zeros(MOTOR_NUM)
        self.tauDes = np.zeros(MOTOR_NUM)
        self.kpDes = np.zeros(MOTOR_NUM)
        self.kdDes = np.zeros(MOTOR_NUM)

class FingerData:
    def __init__(self):
        self.zero()

    def zero(self):
        self.q = np.zeros(MOTOR_NUM)
        self.qd = np.zeros(MOTOR_NUM)
        self.cur = np.zeros(MOTOR_NUM)