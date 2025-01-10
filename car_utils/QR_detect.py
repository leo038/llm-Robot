import cv2
import numpy as np
import pyrealsense2 as rs
from dt_apriltags import Detector
import logging
try:
    from .tools import rotationMatrixToAngles
except:
    from tools import rotationMatrixToAngles

camera_params = [603.39209604, 603.68529217, 334.68581133, 255.07189437]  ##

image_center_pixel = [320, 240, 1]


class QR_Detector():
    def __init__(self):
        self.tag_size = 0.08

        self.at_detector = Detector(searchpath=['apriltags'],
                                    families='tag36h11',
                                    nthreads=1,
                                    quad_decimate=2.0,
                                    quad_sigma=0.0,
                                    refine_edges=1,
                                    decode_sharpening=0.25,
                                    debug=0)

        self.pipeline = rs.pipeline()
        self.realsense_config = rs.config()
        self.realsense_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    def start_camera_stream(self):
        # 启动相机流
        self.pipeline.start(self.realsense_config)

    def stop_camera_stream(self):
        self.pipeline.stop()

    def detect_pose_by_QR_code(self):
        detect_pose = self.detect_pose_by_QR_code_once()
        while detect_pose is None:
            logging.warning(f"未定位成功， 继续定位")
            detect_pose = self.detect_pose_by_QR_code_once()
        return detect_pose

    def detect_pose_by_QR_code_once(self):
        detect_pose = None

        logging.warning(f"开始拉流")
        frames = self.pipeline.wait_for_frames()
        logging.warning(f"结束拉流")
        logging.warning(f"frames: {frames}")
        color_frame = frames.get_color_frame()
        logging.warning(f"获取到的图像： {color_frame}")

        # pipeline.stop()

        if not color_frame:
            logging.error(f"未获取到有效图像")
            return detect_pose

        color_image = np.asanyarray(color_frame.get_data())

        # cv2.namedWindow('QR_detect', flags=cv2.WINDOW_NORMAL |
        #                                    cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
        #
        # cv2.imshow("QR_detect", color_image)
        # cv2.waitKey(30)

        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        logging.warning(f"zhixingjiance")

        tags = self.at_detector.detect(gray, estimate_tag_pose=True, camera_params=camera_params,
                                       tag_size=self.tag_size)
        logging.warning(f"jieshujiance. {tags}")

        if len(tags) >= 1:  ## 可能没有检测到

            H = tags[0].homography
            # print(f"单应性矩阵： {H}")
            corners = tags[0].corners
            # print(f"角点：{corners}")

            # cv2.destroyAllWindows()

            # k = cv2.waitKey(100) & 0xFF
            # if k == ord('s'):
            tmp = np.array(image_center_pixel).reshape(3, -1)
            coord_in_tag = np.dot(np.linalg.inv(H), tmp) * self.tag_size / 2
            # print(f"图像中心在tag坐标系下的坐标：{coord_in_tag}")

            pose_R = tags[0].pose_R

            rotate = rotationMatrixToAngles(pose_R) * -1  ## 相机相对于二维码中心的旋转与二维码相对于相机中心的旋转是反方向的

            ##底盘逆时针旋转， 角度变大， 顺时针旋转， 角度变小
            ##距离上只需要前进， 无需后退

            # print(f"旋转角：{list(rotate)}\n\n")

            detect_pose = {"x": float(coord_in_tag[0]),
                           "y": float(coord_in_tag[1]),
                           "yaw": rotate[-1]  # 只关心绕z轴的旋转
                           }

        logging.warning(f"检测到的位姿：{detect_pose}")

        return detect_pose


if __name__ == "__main__":

    detector = QR_Detector()
    detector.start_camera_stream()
    while True:
        detector.detect_pose_by_QR_code()
