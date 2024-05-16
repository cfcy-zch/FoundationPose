# Python 2/3 compatibility imports
from __future__ import print_function
from six.moves import input

import numpy as np
import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from math import pi
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
from tf import transformations as tf

def homogeneous_matrix_to_pose(matrix):
    # Extract translation vector from homogeneous matrix
    translation = tf.translation_from_matrix(matrix)
    # Extract quaternion from rotation part of homogeneous matrix
    quaternion = tf.quaternion_from_matrix(matrix)
    
    pose_msg = geometry_msgs.msg.Pose()
    pose_msg.position = geometry_msgs.msg.Point(*translation)
    pose_msg.orientation = geometry_msgs.msg.Quaternion(*quaternion)
    
    return pose_msg

moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node('move_group_python_interface_tutorial', anonymous=True)
robot = moveit_commander.RobotCommander()
scene = moveit_commander.PlanningSceneInterface()
group_name = "manipulator_i5"
move_group = moveit_commander.MoveGroupCommander(group_name)
display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                               moveit_msgs.msg.DisplayTrajectory,
                                               queue_size=20)

#添加底部平面
plane_pose = geometry_msgs.msg.PoseStamped()
plane_pose.header.frame_id = "base_link"
plane_pose.pose.orientation.w = 1.0
plane_pose.pose.position.z = -0.05 # 平面上表面与base_link原点重合
plane_name = "plane"
scene.add_box(plane_name, plane_pose, size=(2, 2, 0.1))
#添加灵巧手碰撞区并固连在末端坐标系
hand_heat_box_pose = geometry_msgs.msg.PoseStamped()
hand_heat_box_pose.header.frame_id = "wrist3_Link"
hand_heat_box_pose.pose.orientation.w = 1.0
hand_heat_box_pose.pose.position.x = 0.046387 
hand_heat_box_pose.pose.position.z = 0.00307602133
hand_heat_box_pose.pose.position.z = 0.265/2 # 平面上表面与base_link原点重合
hand_heat_box_name = "hand_heat_box"
scene.add_box(hand_heat_box_name, hand_heat_box_pose, size=(0.2, 0.11, 0.265))
scene.attach_box("wrist3_Link", hand_heat_box_name)
# 我们可以获得该机器人参考坐标系的名字：
planning_frame = move_group.get_planning_frame()
print("============ Planning frame: %s" % planning_frame)

# 我们还可以为该 planning group 打印末端执行器 link 的名字：
eef_link = move_group.get_end_effector_link()
print("============ End effector link: %s" % eef_link)

# 我们可以获得机器人中所有 planning group 的列表:
group_names = robot.get_group_names()
print("============ Available Planning Groups:", robot.get_group_names())

# 有时输出机械臂的所有状态对于调试很有用:
print("============ Printing robot state")
print(robot.get_current_state())
print("")

# # 我们可以从 planning group 中获取并调整各关节角度:
# joint_goal = move_group.get_current_joint_values()
# joint_goal[0] = pi/4
# joint_goal[1] = pi/6
# joint_goal[2] = -pi/4
# joint_goal[3] = 0
# joint_goal[4] = 0
# joint_goal[5] = -pi/2
# # joint_goal[6] = 0

# # 使用关节值或位姿来调用 go 命令，
# # 在已经设置了 planning group 的目标位姿或或目标关节角度的情况下可以不带任何参数。
# move_group.go(joint_goal, wait=True)
# #以base_link作为参考坐标系
move_group.set_pose_reference_frame('base_link')
# # 调用 ``stop()`` 以确保没有未完成的运动。
move_group.stop()

# pose_goal = homogeneous_matrix_to_pose(np.array([[-1, 0, 0, 0.4],
#                                                 [0, -1, 0, 0.2],
#                                                 [0, 0, 1, 0.4],
#                                                 [0, 0, 0, 1]]))
# print(pose_goal)
pose_goal = geometry_msgs.msg.Pose()
pose_goal.orientation.x = 0.612321
pose_goal.orientation.y = 0.35363
pose_goal.orientation.z = 0.182997
pose_goal.orientation.w = 0.683034
pose_goal.position.x = -0.3186
pose_goal.position.y = -0.623452
pose_goal.position.z = 0.599253

move_group.set_pose_target(pose_goal)
plan = move_group.go(wait=True)
print('plan',plan)
# 调用 `stop()` 以确保没有未完成的运动。
move_group.stop()
# 对某一目标位姿进行运动规划以后，最好清除这个目标位姿。
# 注意: 没有类似于 clear_joint_value_targets() 的函数
move_group.clear_pose_targets()

pose_goal_1 = homogeneous_matrix_to_pose(np.array([[-1, 0, 0, 0.4],
                                                    [0, -1, 0, 0.2],
                                                    [0, 0, 1, 0.4],
                                                    [0, 0, 0, 1]]))
move_group.set_pose_target(pose_goal_1)
plan_1 = move_group.go(wait=True)
print('plan_1',plan_1)
move_group.stop()
move_group.clear_pose_targets()

pose_goal_2 = homogeneous_matrix_to_pose(np.array([[1, 0, 0, 0.4],
                                                    [0, 1, 0, 0.2],
                                                    [0, 0, 1, 0.4],
                                                    [0, 0, 0, 1]]))
move_group.set_pose_target(pose_goal_2)
plan_2 = move_group.go(wait=True)
print('plan_2',plan_2)
move_group.stop()
move_group.clear_pose_targets()