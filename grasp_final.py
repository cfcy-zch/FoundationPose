from __future__ import print_function
from estimater import *
from datareader import *
import argparse
import torch, gc
import cv2
import numpy as np
import os
import time
import depthai as dai
import open3d as o3d
from grasp_library import *
import pickle

from six.moves import input
import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from math import pi
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
start_time = time.time()
#foundationpose 初始化
#region
parser = argparse.ArgumentParser()
code_dir = os.path.dirname(os.path.realpath(__file__))
parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/my_bottle/mesh/textured_simple.obj')
parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/my_bottle')
parser.add_argument('--est_refine_iter', type=int, default=5)
parser.add_argument('--track_refine_iter', type=int, default=2)
parser.add_argument('--debug', type=int, default=1)
parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
args = parser.parse_args()

set_logging_format()
set_seed(0)

mesh = trimesh.load(args.mesh_file)

debug = args.debug
debug_dir = args.debug_dir
os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

scorer = ScorePredictor()
refiner = PoseRefinePredictor()
glctx = dr.RasterizeCudaContext()
est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
logging.info("estimator initialization done")

reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=3)  #default: shorter_side = None 如果爆显存把这个值设为400 深度最大值设为3m
#endregion

#读取抓取数据
file_path_1 = "/home/hh/ros1_ws/src/Q-mat for grasp python/file folder/All_Slice.pkl"
file_path_2 = "/home/hh/ros1_ws/src/Q-mat for grasp python/file folder/All_Grasp.pkl"
with open(file_path_1, 'rb') as file:
    All_Slice = pickle.load(file)
with open(file_path_2, 'rb') as file:
    All_Grasp = pickle.load(file)

#moveit commander  move_grounp 初始化
#region
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
    
#endregion

#oak相机初始化
#region
def resize_image(image, scale_percent):
    """
    调整图像大小并返回调整后的图像。

    Args:
        image: 要调整大小的图像。
        scale_percent: 缩放百分比（0-100）。

    Returns:
        调整大小后的图像。
    """
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized_image
COLOR = True

lrcheck = True  # Better handling for occlusions
extended = False  # Closer-in minimum depth, disparity range is doubled
subpixel = True  # Better accuracy for longer distance, fractional disparity 32-levels
# Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7
median = dai.StereoDepthProperties.MedianFilter.KERNEL_7x7

print("StereoDepth config options:")
print("    Left-Right check:  ", lrcheck)
print("    Extended disparity:", extended)
print("    Subpixel:          ", subpixel)
print("    Median filtering:  ", median)

pipeline = dai.Pipeline()

monoLeft = pipeline.create(dai.node.MonoCamera)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)

monoRight = pipeline.create(dai.node.MonoCamera)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

stereo = pipeline.createStereoDepth()
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.initialConfig.setMedianFilter(median)
# stereo.initialConfig.setConfidenceThreshold(255)

stereo.setLeftRightCheck(lrcheck)
stereo.setExtendedDisparity(extended)
stereo.setSubpixel(subpixel)
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

config = stereo.initialConfig.get()
config.postProcessing.speckleFilter.enable = False
config.postProcessing.speckleFilter.speckleRange = 50
config.postProcessing.temporalFilter.enable = True
config.postProcessing.spatialFilter.enable = True
config.postProcessing.spatialFilter.holeFillingRadius = 2
config.postProcessing.spatialFilter.numIterations = 1
config.postProcessing.thresholdFilter.minRange = 400
config.postProcessing.thresholdFilter.maxRange = 200000
config.postProcessing.decimationFilter.decimationFactor = 1
stereo.initialConfig.set(config)

xout_depth = pipeline.createXLinkOut()
xout_depth.setStreamName("depth")
stereo.depth.link(xout_depth.input)

# xout_disparity = pipeline.createXLinkOut()
# xout_disparity.setStreamName('disparity')
# stereo.disparity.link(xout_disparity.input)

xout_colorize = pipeline.createXLinkOut()
xout_colorize.setStreamName("colorize")
xout_rect_left = pipeline.createXLinkOut()
xout_rect_left.setStreamName("rectified_left")
xout_rect_right = pipeline.createXLinkOut()
xout_rect_right.setStreamName("rectified_right")

if COLOR:
    camRgb = pipeline.create(dai.node.ColorCamera)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setIspScale(1, 3)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
    camRgb.initialControl.setManualFocus(130)
    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
    camRgb.isp.link(xout_colorize.input)
else:
    stereo.rectifiedRight.link(xout_colorize.input)

stereo.rectifiedLeft.link(xout_rect_left.input)
stereo.rectifiedRight.link(xout_rect_right.input)


class HostSync:
    def __init__(self):
        self.arrays = {}

    def add_msg(self, name, msg):
        if not name in self.arrays:
            self.arrays[name] = []
        # Add msg to array
        self.arrays[name].append({"msg": msg, "seq": msg.getSequenceNum()})

        synced = {}
        for name, arr in self.arrays.items():
            for i, obj in enumerate(arr):
                if msg.getSequenceNum() == obj["seq"]:
                    synced[name] = obj["msg"]
                    break
        # If there are 5 (all) synced msgs, remove all old msgs
        # and return synced msgs
        if len(synced) == 4:  # color, left, right, depth, nn
            # Remove old msgs
            for name, arr in self.arrays.items():
                for i, obj in enumerate(arr):
                    if obj["seq"] < msg.getSequenceNum():
                        arr.remove(obj)
                    else:
                        break
            return synced
        return False


with dai.Device(pipeline) as device:

    device.setIrLaserDotProjectorBrightness(1200)
    qs = []
    qs.append(device.getOutputQueue("depth", maxSize=1, blocking=False))
    qs.append(device.getOutputQueue("colorize", maxSize=1, blocking=False))
    qs.append(device.getOutputQueue("rectified_left", maxSize=1, blocking=False))
    qs.append(device.getOutputQueue("rectified_right", maxSize=1, blocking=False))

    try:
        from projector_3d import PointCloudVisualizer
    except ImportError as e:
        raise ImportError(
            f"\033[1;5;31mError occured when importing PCL projector: {e}. Try disabling the point cloud \033[0m "
        )

    calibData = device.readCalibration()
    if COLOR:
        w, h = camRgb.getIspSize()
        intrinsics = calibData.getCameraIntrinsics(dai.CameraBoardSocket.RGB, dai.Size2f(w, h))
    else:
        w, h = monoRight.getResolutionSize()
        intrinsics = calibData.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT, dai.Size2f(w, h))
    # pcl_converter = PointCloudVisualizer(intrinsics, w, h)

    serial_no = device.getMxId()
    sync = HostSync()
    depth_vis, color, rect_left, rect_right = None, None, None, None
    pTime = 0
    cTime = 0
#endregion
    
    #主循环开始
    # i = 0
    j = 0
    time.sleep(10)
    end_time = time.time()
    execution_time = end_time - start_time
    print("初始化时间：", execution_time, "秒")
    while j == 0:
        # if i == 0:
        # 加载图像
        for q in qs:
            new_msg = q.tryGet()
            if new_msg is not None:
                msgs = sync.add_msg(q.getName(), new_msg)
                if msgs:
                    image = msgs["colorize"].getCvFrame()
                    depth = msgs["depth"].getFrame()
                    i = 1
                    # 创建窗口并显示图像
                    cv2.namedWindow('image')
                    cv2.imshow('image', image)
                    color = reader.get_color_oak(image)
                    # 定义变量以保存矩形的坐标
                    rect_start = None
                    rect_end = None
                    drawing = False
                    save_mask = None
                    # 鼠标事件回调函数
                    def draw_rect(event, x, y, flags, param):
                        global rect_start, rect_end, drawing, save_mask

                        if event == cv2.EVENT_LBUTTONDOWN:
                            drawing = True
                            rect_start = (x, y)

                        elif event == cv2.EVENT_MOUSEMOVE:
                            if drawing:
                                rect_end = (x, y)
                                # 画矩形
                                temp_image = image.copy()
                                cv2.rectangle(temp_image, rect_start, rect_end, (0, 255, 0), 2)
                                cv2.imshow('image', temp_image)

                        elif event == cv2.EVENT_LBUTTONUP:
                            drawing = False
                            rect_end = (x, y)
                            # 画最终矩形
                            cv2.rectangle(image, rect_start, rect_end, (0, 255, 0), 2)
                            cv2.imshow('image', image)
                            # 生成掩码
                            mask = generate_mask(image.shape[:2], rect_start, rect_end)
                            cv2.imshow('mask', mask)
                            # 等待用户按键确认或取消
                            key = cv2.waitKey(0) & 0xFF
                            if key == ord('s'):  # 确认操作
                                # 定义相对路径
                                output_dir = 'test_folder'
                                output_path = os.path.join(output_dir, 'mask_image.png')
                                # cv2.imwrite(output_path, mask)
                                save_mask = mask
                            elif key == ord('c'):  # 取消操作
                                # 清除掩码窗口
                                cv2.destroyWindow('mask')
                                # 重置绘制矩形的状态
                                drawing = False
                                rect_start = None
                                rect_end = None
                                # 清除图像上的矩形框
                                cv2.imshow('image', image)

                    # 生成掩码函数
                    def generate_mask(shape, start_point, end_point):
                        mask = np.zeros(shape, dtype=np.uint8)
                        cv2.rectangle(mask, start_point, end_point, (255), -1)
                        return mask

                    # 设置鼠标事件回调函数
                    cv2.setMouseCallback('image', draw_rect)
                    # 等待用户操作
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    depth = reader.get_depth_oak(depth)
                    mask = reader.get_mask_oak(save_mask)#.astype(bool)
                    pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
                    if debug>=3:
                        m = mesh.copy()
                        m.apply_transform(pose)
                        m.export(f'{debug_dir}/model_tf.obj')
                        xyz_map = depth2xyzmap(depth, reader.K)
                        valid = depth>=0.1
                        pcd = toOpen3dCloud(xyz_map[valid], color[valid])
                        o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
                    if debug>=1:
                        center_pose = pose#@np.linalg.inv(to_origin)
                        vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose@np.linalg.inv(to_origin), bbox=bbox)
                        vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
                        cv2.imshow('1', vis[...,::-1])
                        cv2.waitKey(1)
                    if debug>=2:
                        os.makedirs(f'{reader.video_dir}/track_vis', exist_ok=True)
                        # imageio.imwrite(f'{reader.video_dir}/track_vis/{reader.id_strs[i]}.png', vis)
                    # print(type(pose.reshape(4, 4)))#<class 'numpy.ndarray'>
                    cv2.waitKey(0)
                    #筛选
                    H_base_cam = rpy_d_to_H([2.5437, 0.67976, 0.74532],[0.017891, -0.30749, 1.6099])#后面要改一次
                    H_obj_cam = pose.reshape(4, 4)
                    # H_hand_obj = H_list
                    H_hand_tool = rpy_d_to_H ([-1.5708, -0.61087, -0.055182],[0.026734, 0.00824, 0.10575])
                    for grasp_name, grasp_obj in All_Grasp.items():
                        grasp_obj.get_final_H(H_cam_base = homogeneous_inverse(H_base_cam), H_obj_cam = H_obj_cam, H_tool_hand = homogeneous_inverse(H_hand_tool))
                    you_like = []
                    for key, value in All_Grasp.items():
                        #机械手坐标系高度要够 值要改一次
                        indices = [i for i, matrix in enumerate(value.H_list_hand) if matrix[2, 3] > 0.11/2]
                        if indices :#and value.slice.shape == 'Line' and value.slice.edgepoint_id is not None:
                            for idx in indices:
                                you_like.append((key, idx))
                    while True:
                        user_inputnum = input(f"总计{len(you_like)}个抓取，请输入抓取编号0~{len(you_like)-1}查看或输入超范围数值跳过: ")
                        which_one = int(user_inputnum)
                        if 0 <= which_one <= len(you_like) - 1:
                            All_Grasp[you_like[which_one][0]].show_which_grasp(you_like[which_one][1])
                            user_input = input("按下 's' 键以确认当前抓取并进行机械臂操作,按下 'q' 键直接跳出本轮抓取,按其余键重新选择抓取: ")
                            if user_input.lower() == 's':
                                #以base_link作为参考坐标系
                                move_group.set_pose_reference_frame('base_link')
                                #起始位姿
                                move_group.stop()
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
                                move_group.stop()
                                move_group.clear_pose_targets()# 对某一目标位姿进行运动规划以后，最好清除这个目标位姿。
                                #hand_reday位姿
                                H_translation = np.array([[1, 0, 0, -0.14],              # 平移一个距离
                                                        [0, 1, 0, 0],
                                                        [0, 0, 1, 0],
                                                        [0, 0, 0, 1]
                                                        ])
                                pose_goal_1 = homogeneous_matrix_to_pose(np.array([[-1, 0, 0, 0.4],
                                                                                    [0, -1, 0, 0.2],
                                                                                    [0, 0, 1, 0.4],
                                                                                    [0, 0, 0, 1]]))
                                move_group.set_pose_target(pose_goal_1)
                                plan_1 = move_group.go(wait=True)
                                print('plan_1',plan_1)
                                move_group.stop()
                                move_group.clear_pose_targets()
                                #hand_final位姿
                                pose_goal_2 = homogeneous_matrix_to_pose(np.array([[1, 0, 0, 0.4],
                                                                                    [0, 1, 0, 0.2],
                                                                                    [0, 0, 1, 0.4],
                                                                                    [0, 0, 0, 1]]))
                                move_group.set_pose_target(pose_goal_2)
                                plan_2 = move_group.go(wait=True)
                                print('plan_2',plan_2)
                                move_group.stop()
                                move_group.clear_pose_targets()
                                break
                            elif user_input.lower() == 'q':
                                break
                            else:#可以按c
                                print (f'再次选取抓取，当前抓取编号{which_one}')
                        else : 
                            break
                    # if you_like:
                    #     for which_one in you_like:
                    #         All_Grasp[which_one[0]].show_which_grasp(which_one[1])
                    #         user_input = input("按下 's' 键以继续: ")
                    #         if user_input.lower() == 's':
                    #             print("继续")
                    #             break

                    # 清除GPU缓存
                    # torch.cuda.empty_cache()
                    # All_Grasp['Grasp1'].show_grasp()
                    while True:
                        #oak相机显示
                        for q in qs:
                            new_msg = q.tryGet()
                            if new_msg is not None:
                                msgs = sync.add_msg(q.getName(), new_msg)
                                if msgs:
                                    depth = msgs["depth"].getFrame()
                                    color = msgs["colorize"].getCvFrame()
                                    # rectified_left = msgs["rectified_left"].getCvFrame()
                                    # rectified_right = msgs["rectified_right"].getCvFrame()
                                    # print(depth.shape)
                                    # print(color.shape)
                                    # print(color.dtype)
                                    # print(depth.dtype)
                                    # print("OpenCV版本:", cv2.__version__)
                                    depth_format = depth#(depth/1000).astype(np.uint8)
                                    # depth_vis = cv2.normalize(depth, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
                                    # depth_vis = cv2.equalizeHist(depth_vis)
                                    # depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_HOT)
                                    # depth_format_vis = cv2.equalizeHist(depth_format)
                                    # depth_format_vis = cv2.applyColorMap(depth_format, cv2.COLORMAP_JET)
                                    # cv2.imshow("depth_format", depth_format_vis)
                                    # cv2.imshow("depth", depth_vis)#resize_image(depth_vis,30))
                                    # cv2.imshow("color", color)#resize_image(color,30))
                                    # cv2.imshow("rectified_left", rectified_left)
                                    # cv2.imshow("rectified_right", rectified_right)
                                    # rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
                                    # pcl_converter.rgbd_to_projection(depth, rgb)
                                    # pcl_converter.visualize_pcd()
                                    color = reader.get_color_oak(color)
                                    depth = reader.get_depth_oak(depth)
                                    pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=args.track_refine_iter)
                                    center_pose = pose#@np.linalg.inv(to_origin)
                                    vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose@np.linalg.inv(to_origin), bbox=bbox)
                                    vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
                                    # cTime = time.time()
                                    # fps = 1 / (cTime - pTime)       #计算处理频率
                                    # pTime = cTime
                                    # #rate.sleep()
                                    # cv2.putText(vis, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,(255, 255, 255), 2)
                                    cv2.imshow('1', vis[...,::-1])

                        key = cv2.waitKey(1)
                        if key == ord("s"):
                            timestamp = str(int(time.time()))
                            print('重新开始一轮')
                            # cv2.imwrite(f"{serial_no}_{timestamp}_depth_color.png", depth_vis)
                            # cv2.imwrite(f"{serial_no}_{timestamp}_depth.png", depth_format)
                            # cv2.imwrite(f"{serial_no}_{timestamp}_color.png", color)
                            # cv2.imwrite(f"{serial_no}_{timestamp}_rectified_left.png", rectified_left)
                            # cv2.imwrite(f"{serial_no}_{timestamp}_rectified_right.png", rectified_right)
                            # o3d.io.write_point_cloud(f"{serial_no}_{timestamp}.pcd", pcl_converter.pcl, compressed=True)
                            break
                        elif key == ord("q"):
                            j = 1
                            print('退出')
                            break