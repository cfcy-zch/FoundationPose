import numpy as np
from grasp_library import *

def rot(cta_rad, x):
    if x == 1:
        A = np.array([[1, 0, 0],
                      [0, np.cos(cta_rad), -np.sin(cta_rad)],
                      [0, np.sin(cta_rad), np.cos(cta_rad)]])
    elif x == 2:
        A = np.array([[np.cos(cta_rad), 0, np.sin(cta_rad)],
                      [0, 1, 0],
                      [-np.sin(cta_rad), 0, np.cos(cta_rad)]])
    elif x == 3:
        A = np.array([[np.cos(cta_rad), -np.sin(cta_rad), 0],
                      [np.sin(cta_rad), np.cos(cta_rad), 0],
                      [0, 0, 1]])
    else:
        raise ValueError("x must be 1, 2, or 3")
    return A
#给出rpy和d返回H
def rpy_d_to_H(rpy, d):
    rot_matrix = rot(rpy[2], 3)@rot(rpy[1], 2)@rot(rpy[0], 1)
    H = np.eye(4)
    H[:3, :3] = rot_matrix
    H[0,3] = d[0]
    H[1,3] = d[1]
    H[2,3] = d[2]
    return H

# 示例用法

result = rpy_d_to_H([2.5437,0.67976,0.74532],[0,1,2])
print(result)
pose_goal = homogeneous_matrix_to_pose(np.array([[1, 0, 0, 0.4],
                                                [0, 1, 0, 0],
                                                [0, 0, 1, 0.16],
                                                [0, 0, 0, 1]]))
print(pose_goal)