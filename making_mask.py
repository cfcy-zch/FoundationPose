import cv2
import numpy as np
import os

# 加载图像
image = cv2.imread('/home/hh/ros1_ws/src/FoundationPose/1944301031DCA12E00_1714962714_color.png')

# 创建窗口并显示图像
cv2.namedWindow('image')
cv2.imshow('image', image)

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
            cv2.imwrite(output_path, mask)
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
# cv2.imshow('save_mask', save_mask)
# cv2.waitKey(0)
