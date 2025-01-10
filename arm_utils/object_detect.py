#!/usr/bin/env python3
# 导入依赖
# text recognition
import logging
import time
from copy import deepcopy

import cv2
import numpy as np
import pyrealsense2 as rs
import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

from .crnn import CRNN
from .str_process import strLabelConverter


class ObjectDetect():
    def __init__(self):
        self.detect_model = YOLO(
            '/home/nvidia/leo/robot/robot_sdk/custom_plugins/src/realman_plugins/realman_plugins/utils/asserts/yolov8n_last_elevator_button.pt',
            verbose=False)  # 加载YoloV8模型
        # load model parameters
        self.img_resize = (32, 32)
        self.alphabet = '0123456789GBLM()$C!*<>'

        self.reg_model = CRNN(self.img_resize[1], len(self.alphabet) + 1)
        if torch.cuda.is_available():
            self.reg_model = self.reg_model.cuda()
        self.reg_model.load_state_dict(
            torch.load(
                "/home/nvidia/leo/robot/robot_sdk/custom_plugins/src/realman_plugins/realman_plugins/utils/asserts/elevator_crnn_3_32_100_lanrun_huoti.pth"))
        # map_location=torch.device('cpu')))
        self.reg_model.eval()
        logging.info("[INFO] 完成YoloV8模型加载")

        self.target_class_name_list = []
        self.target_pose_list = []
        self.reserved_target_num = 10

        # self.is_start_camera = False

        # self.detect_objects()

        # 配置 RealSense
        self.pipeline = rs.pipeline()
        self.realsense_config = rs.config()
        self.realsense_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
        self.realsense_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)

    def start_camera_stream(self):
        # 启动相机流
        self.pipeline.start(self.realsense_config)

    def stop_camera_stream(self):
        self.pipeline.stop()

    def _update_object(self, class_name, target_pose):
        # logging.warning(f"检测的目标信息： {class_name, target_pose}")
        self.target_class_name_list = self.target_class_name_list[-1 * self.reserved_target_num:]
        self.target_pose_list = self.target_pose_list[-1 * self.reserved_target_num:]

        self.target_class_name_list.append(class_name)
        self.target_pose_list.append(target_pose)

    def find_right_target(self, object_class_name):

        target_name_list = deepcopy(self.target_class_name_list)  ##拷贝一份， 避免在过程中信息被更新， 导致错乱
        target_pose_list = deepcopy(self.target_pose_list)
        final_target_pose = None
        for idx in range(len(target_name_list) - 1, -1, -1):  ##从列表最后往前找， 保证取最新的
            if object_class_name == target_name_list[idx]:
                final_target_pose = target_pose_list[idx]
                break
        return final_target_pose

    def get_aligned_images(self):

        align_to = rs.stream.color  # 与color流对齐
        align = rs.align(align_to)

        frames = self.pipeline.wait_for_frames()  # 等待获取图像帧
        aligned_frames = align.process(frames)  # 获取对齐帧
        aligned_depth_frame = aligned_frames.get_depth_frame()  # 获取对齐帧中的depth帧
        color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的color帧

        # 相机参数的获取
        intr = color_frame.profile.as_video_stream_profile().intrinsics  # 获取相机内参
        depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics  # 获取深度参数

        depth_image = np.asanyarray(aligned_depth_frame.get_data())  # 深度图（默认16位）
        color_image = np.asanyarray(color_frame.get_data())  # RGB图

        # 返回相机内参、深度参数、彩色图、深度图、齐帧中的depth帧
        return intr, depth_intrin, color_image, depth_image, aligned_depth_frame

    def get_3d_camera_coordinate(self, depth_pixel, aligned_depth_frame, depth_intrin):
        x = depth_pixel[0]
        y = depth_pixel[1]
        dis = aligned_depth_frame.get_distance(x, y)  # 获取该像素点对应的深度

        # print(f"深度值：{x, y, dis}")
        # print(f"depth intrin:{depth_intrin}")
        camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, dis)
        # print(f"相机坐标：", camera_coordinate)
        return dis, camera_coordinate

    def detect_corner_once(self, corner_pose=[0, 0]):
        parten_size = (5, 4)

        intr, depth_intrin, color_image, depth_image, aligned_depth_frame = self.get_aligned_images()

        if not depth_image.any() or not color_image.any():
            return
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, parten_size, None)
        if ret:
            roi_pos = corner_pose[0] * parten_size[0] + corner_pose[1]
            print(corners.shape)
            print(f"左上角点：{corners[0, 0]}")
            print(f"右下角点：{corners[-1, -1]}")

            ux, uy = corners[roi_pos, 0]

            # print(f"原始深度值：{depth_image[int(ux), int(uy)]}")
            dis, camera_coordinate = self.get_3d_camera_coordinate([ux, uy], aligned_depth_frame,
                                                                   depth_intrin)  # 得到中心点的深度值,物体坐标

            text = f"{camera_coordinate[0]:.2f}, {camera_coordinate[1]:.2f}, {camera_coordinate[2]:.2f}"
            cv2.putText(color_image, str(text), (int(ux) + 10, int(uy) + 10), 0, 1,
                        [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
            # 绘制角点并显示图像
            cv2.drawChessboardCorners(color_image, parten_size, corners, ret)

            self._update_object(class_name="corner", target_pose=camera_coordinate)

        cv2.imshow('Chessboard', color_image)
        cv2.waitKey(1)  ## 停留1s, 观察找到的角点是否正确

    def detect_objects(self):

        self.start_camera_stream()
        for i in range(10):
            self.detect_objects_once()
            time.sleep(0.01)

        self.stop_camera_stream()
        logging.warning(f"目标检测完成， 关闭相机流。")
        cv2.destroyAllWindows()

    def detect_corner(self, corner_pose=[0, 0]):
        self.start_camera_stream()
        for i in range(30):
            self.detect_corner_once(corner_pose=corner_pose)
            time.sleep(0.01)

        self.stop_camera_stream()
        logging.warning(f"角点检测完成， 关闭相机流。")
        cv2.destroyAllWindows()

    def detect_objects_once(self):
        # 获取图像流
        intr, depth_intrin, color_image, depth_image, aligned_depth_frame = self.get_aligned_images()

        if not depth_image.any() or not color_image.any():
            return

        # 使用 YOLOv8 进行目标检测
        results = self.detect_model.predict(color_image, conf=0.5, )
        detected_boxes = results[0].boxes.xyxy  # 获取边界框坐标
        data = results[0].boxes.data.cpu().tolist()
        canvas = results[0].plot()

        u_x_center = 320
        u_y_center = 240
        cv2.circle(canvas, (u_x_center, u_y_center), 4, (0, 0, 255), 5)
        cv2.putText(canvas, "0,0", (u_x_center + 20, u_y_center + 10), 0, 1,
                    [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)

        for i, (row, box) in enumerate(zip(data, detected_boxes)):
            id = int(row[5])
            name = results[0].names[id]
            x1, y1, x2, y2 = map(int, box)  # 获取边界框坐标

            # some args
            # data preprocesser
            transformer = transforms.Compose([transforms.Resize(self.img_resize), transforms.ToTensor()])
            converter = strLabelConverter(self.alphabet)

            # recognition
            patch = color_image[y1:y2, x1:x2]
            patch = Image.fromarray(patch)

            patch = transformer(patch)

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            preds = self.reg_model(patch.unsqueeze(0).cuda())

            end.record()
            torch.cuda.synchronize()
            elapsed_time = start.elapsed_time(end)

            # logging.warning(f" 帧率： {1 / elapsed_time*1000}")
            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            preds_size = torch.IntTensor([preds.size(0)])
            class_name = converter.decode(preds.data, preds_size.data, raw=False)

            # # log the results
            # print(sim_pred)
            #
            # print(f"bbox: {x1, y1, x2, y2}")
            # 显示中心点坐标
            ux = int((x1 + x2) / 2)
            uy = int((y1 + y2) / 2)
            dis, camera_coordinate = self.get_3d_camera_coordinate([ux, uy], aligned_depth_frame,
                                                                   depth_intrin)  # 得到中心点的深度值,物体坐标

            formatted_camera_coordinate = f"({camera_coordinate[0]:.2f}, {camera_coordinate[1]:.2f}, {camera_coordinate[2]:.2f})"
            # # 展示检测界面
            cv2.circle(canvas, (ux, uy), 4, (255, 255, 255), 5)
            cv2.putText(canvas, str(formatted_camera_coordinate), (ux + 20, uy + 10), 0, 1,
                        [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)

            cv2.putText(canvas, str(class_name), (ux - 10, uy - 10), 0, 1,
                        [0, 0, 255], thickness=2, lineType=cv2.LINE_AA)

            self._update_object(class_name=class_name, target_pose=camera_coordinate)

        cv2.namedWindow('detection', flags=cv2.WINDOW_NORMAL |
                                           cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
        cv2.imshow('detection', canvas)
        key = cv2.waitKey(30)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
        #
        # time1 = Timer(0.2, self.detect_objects)  ## 最高数据帧率为50
        # time1.start()


def main(args=None):
    pass


if __name__ == '__main__':
    main()
