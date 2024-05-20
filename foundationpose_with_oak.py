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

reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=3)  #default: shorter_side = None 如果爆显存把这个值设为400
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
    i = 0
    time.sleep(10)
    end_time = time.time()
    execution_time = end_time - start_time
    print("初始化时间：", execution_time, "秒")
    while True:
        if i == 0:
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
        else:
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
                        cTime = time.time()
                        fps = 1 / (cTime - pTime)       #计算处理频率
                        pTime = cTime
                        #rate.sleep()
                        cv2.putText(vis, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,(255, 255, 255), 2)
                        cv2.imshow('1', vis[...,::-1])

            key = cv2.waitKey(1)
            if key == ord("s"):
                timestamp = str(int(time.time()))
                # cv2.imwrite(f"{serial_no}_{timestamp}_depth_color.png", depth_vis)
                # cv2.imwrite(f"{serial_no}_{timestamp}_depth.png", depth_format)
                # cv2.imwrite(f"{serial_no}_{timestamp}_color.png", color)
                # cv2.imwrite(f"{serial_no}_{timestamp}_rectified_left.png", rectified_left)
                # cv2.imwrite(f"{serial_no}_{timestamp}_rectified_right.png", rectified_right)
                # o3d.io.write_point_cloud(f"{serial_no}_{timestamp}.pcd", pcl_converter.pcl, compressed=True)
            elif key == ord("q"):
                break
        # i = i + 1