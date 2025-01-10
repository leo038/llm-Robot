import logging
from copy import deepcopy

import numpy as np
from scipy.spatial.transform import Rotation as R

from .hand import HandControl
from .object_detect import ObjectDetect
from .robotic_arm import *

ARM_VELOCITY = 10  ## 机械臂默认移动速度


def convert(target_pose, arm_pose):
    """
    我们需要将旋转向量和平移向量转换为齐次变换矩阵，然后使用深度相机识别到的物体坐标（x, y, z）和
    机械臂末端的位姿（x1,y1,z1,rx,ry,rz）来计算物体相对于机械臂基座的位姿（x, y, z, rx, ry, rz）
    """

    x, y, z = target_pose
    x1, y1, z1, rx, ry, rz = arm_pose
    # 相机坐标系到机械臂末端坐标系的旋转矩阵和平移向量
    # 默认值
    # rotation_matrix = np.array([[0.01206237, 0.99929647, 0.03551135],
    #                             [-0.99988374, 0.01172294, 0.00975125],
    #                             [0.00932809, -0.03562485, 0.9993217]])
    # translation_vector = np.array([-0.08039019, 0.03225555, -0.08256825])

    ##标定板间隔按20mm计算
    # rotation_matrix = np.array([[-0.47732551, 0.86350562, -0.16284471],
    #                             [-0.87863043, -0.46626518, 0.10298224],
    #                             [0.01299693, 0.19223637, 0.98126258]])
    # translation_vector = np.array([-0.03741188, 0.04725729, 0.0614214])

    ##标定板间隔按19mm计算
    #
    # rotation_matrix = np.array([[-0.477326, 0.86350536, -0.16284468],
    #                             [-0.87863017, -0.46626566, 0.10298235],
    #                             [0.01299692, 0.1922364, 0.98126258]])
    # translation_vector = np.array([-0.03525436, 0.05038928, 0.05395732])

    ## 用35mm标定板重新标定后（固定安装的机械臂）
    # rotation_matrix = np.array(
    #     [[-0.47774048, 0.85821521, -0.18769838],
    #      [-0.8784099, -0.46358326, 0.11613187],
    #      [0.01265231, 0.22035701, 0.97533723]])
    # translation_vector = np.array([-0.03907758, 0.05756365, -0.00361838])

    # ## 安装在车上的机械臂
    # rotation_matrix = np.array(
    #     [[-0.485411, 0.85264408, -0.19332415],
    #      [-0.87428577, -0.47320663, 0.108166],
    #      [0.00074483, 0.22152552, 0.97515429]])
    # translation_vector = np.array([-0.04235172, 0.06532541, -0.00554681])

    ## 设置安装位置后

    rotation_matrix = np.array(
        [[-0.50091884, 0.84518061, - 0.18641367],
         [-0.86543987, -0.49154582, 0.09693574],
         [-0.00970265, 0.20988676, 0.97767756]])
    translation_vector = np.array([-0.03964659, 0.0635493, -0.00715257])

    # 深度相机识别物体返回的坐标
    obj_camera_coordinates = np.array([x, y, z])

    # 机械臂末端的位姿，单位为弧度
    end_effector_pose = np.array([x1, y1, z1, rx, ry, rz])

    # 将旋转矩阵和平移向量转换为齐次变换矩阵
    T_camera_to_end_effector = np.eye(4)
    T_camera_to_end_effector[:3, :3] = rotation_matrix
    T_camera_to_end_effector[:3, 3] = translation_vector

    # 机械臂末端的位姿转换为齐次变换矩阵
    position = end_effector_pose[:3]
    orientation = R.from_euler('xyz', end_effector_pose[3:], degrees=False).as_matrix()
    T_base_to_end_effector = np.eye(4)
    T_base_to_end_effector[:3, :3] = orientation
    T_base_to_end_effector[:3, 3] = position

    # 计算物体相对于机械臂基座的位姿
    obj_camera_coordinates_homo = np.append(obj_camera_coordinates, [1])  # 将物体坐标转换为齐次坐标
    obj_end_effector_coordinates_homo = T_camera_to_end_effector.dot(obj_camera_coordinates_homo)
    obj_base_coordinates_homo = T_base_to_end_effector.dot(obj_end_effector_coordinates_homo)
    obj_base_coordinates = obj_base_coordinates_homo[:3]  # 从齐次坐标中提取物体的x, y, z坐标

    # 计算物体的旋转
    obj_orientation_matrix = T_base_to_end_effector[:3, :3].dot(rotation_matrix)
    obj_orientation_euler = R.from_matrix(obj_orientation_matrix).as_euler('xyz', degrees=False)

    # 组合结果
    obj_base_pose = np.hstack((obj_base_coordinates, obj_orientation_euler))
    # obj_base_pose[3:] = rx, ry, rz
    # obj_base_pose[4] = -1.57  # rad  让手掌保持水平   TODO 固定在支架上的机械臂
    # obj_base_pose[4] = -1.45  # rad  让手掌保持水平   TODO  固定在车上的机械臂
    # obj_base_pose[5] = -0.45  # rad  让手掌保持水平   TODO  固定在车上的机械臂
    # obj_base_pose[4] = 1.47     # 不要加末端姿态约束， 每个位置都不一样， 可能导致机械手无法到达
    # obj_base_pose[5] = 1.45
    return obj_base_pose  


class ArmControl():

    def __init__(self):
        self.init_arm()

        self.hand_control = HandControl(arm=self.arm)

        self.object_detector = ObjectDetect()

    def init_arm(self):
        self.arm = Arm(dev_mode=RM75, ip="192.168.1.18")
        logging.info("设置通讯端口 Modbus RTU 模式")
        self.arm.Set_Modbus_Mode(port=1, baudrate=115200, timeout=2)
        logging.info("打开末端24V电源")
        self.arm.Set_Tool_Voltage(type=3)

    def detect_and_catch(self, object_class_name, arm_speed, extra_deep):

        self.object_detector.detect_objects()

        target_pose = self.object_detector.find_right_target(object_class_name)
        if target_pose is None:
            logging.warning(f"未检测到目标信息： {object_class_name}")
            return False

        ret = self.catch_object(target_pose=target_pose, arm_speed=arm_speed, extra_deep=extra_deep)

        return ret



    def detect_corner_and_catch(self, corner_pose, arm_speed, extra_deep):

        self.object_detector.detect_corner(corner_pose=corner_pose)

        object_class_name = "corner"
        target_pose = self.object_detector.find_right_target(object_class_name)
        if target_pose is None:
            logging.warning(f"未检测到目标信息： {object_class_name}")
            return False

        ret = self.catch_object(target_pose=target_pose, arm_speed=arm_speed, extra_deep=extra_deep)

        return ret



    def catch_object(self, target_pose, arm_speed=None, extra_deep= None):

        speed = arm_speed or ARM_VELOCITY
        logging.warning(f"[0]切换到 Arm_Tip 工具坐标系下")
        self.arm.Change_Tool_Frame(name="Arm_Tip")

        # 首先把物体在相机坐标系下位置转到机械臂坐标系下
        _, _, cur_arm_pose, _, _ = self.arm.Get_Current_Arm_State()

        target_pose1 = deepcopy(target_pose)
        target_pose1[2] = target_pose1[2] - 0.05  ## 先让机械手到达深度差6cm的位置

        target_pose2 = deepcopy(target_pose)
        if extra_deep is None:
            target_pose2[2] = target_pose2[2] + 0.013
        else:
            target_pose2[2] = target_pose2[2] + extra_deep

        obj_pose1 = convert(target_pose=target_pose1, arm_pose=cur_arm_pose)
        obj_pose2 = convert(target_pose=target_pose2, arm_pose=cur_arm_pose)

        logging.warning(f"待抓取物体的位置（相机坐标系下）：{target_pose}")
        logging.warning(f"待抓取物体的位置（机械臂基坐标系下）：{obj_pose2[:3]}")

        logging.warning(f"打开食指...")
        self.hand_control.gesture_generate(control_data=[100, 0, 100, 100, 100, 100])  ##打开食指,

        logging.warning(f"切换到 shizhi0924 工具坐标系下")
        self.arm.Change_Tool_Frame(name="shizhi0924")

        logging.warning(f"移动位置1：{obj_pose1}")
        flag = self.arm.Movej_P_Cmd(pose=obj_pose1, v=speed, trajectory_connect=0)
        if flag != 0:
            return False

        time.sleep(0.1)

        logging.warning(f"移动位置2：{obj_pose2}")
        # flag = self.arm.Movel_Cmd(pose=obj_pose2, v=speed, trajectory_connect=0)
        flag = self.arm.Movej_P_Cmd(pose=obj_pose2, v=speed, trajectory_connect=0)
        if flag != 0:
            return False

        logging.warning(f"切换到 Arm_Tip 工具坐标系下")
        self.arm.Change_Tool_Frame(name="Arm_Tip")

        time.sleep(0.1)
        # 恢复
        logging.warning(f"恢复到初始位置：{cur_arm_pose}")
        flag = self.arm.Movej_P_Cmd(pose=cur_arm_pose, v=70, trajectory_connect=0)
        if flag != 0:
            return False

        logging.warning(f"手掌展开...")
        self.hand_control.gesture_generate(control_data=[0, 0, 0, 0, 0, 0])
        logging.warning(f"任务完成！！！")

        return True

    def joints_move(self, joints, arm_speed=None):
        speed = arm_speed or ARM_VELOCITY
        ret = self.arm.Movej_Cmd(joint=joints, v=speed, trajectory_connect=0)
        if ret == 0:
            return True
        else:
            return False

    def cartesian_move(self, cartesian_pose, arm_speed=None, linear=False):
        self.arm.Change_Tool_Frame(name="Arm_Tip")
        speed = arm_speed or ARM_VELOCITY
        if linear:
            ret = self.arm.Movel_Cmd(pose=cartesian_pose, v=speed, trajectory_connect=0)
        else:
            ret = self.arm.Movej_P_Cmd(pose=cartesian_pose, v=speed, trajectory_connect=0)
        if ret == 0:
            return True
        else:
            return False

    def expand_joint_move(self, angular, arm_speed=None):
        speed = arm_speed or ARM_VELOCITY
        ret = self.arm.Expand_Set_Pos(pos=int(1000 * angular), speed=speed, block=True)
        if ret == 0:
            return True
        else:
            return False

    def init_arm_pose(self, arm_speed=None):
        speed = arm_speed or ARM_VELOCITY
        joints = [0, 90, 170, 0, 0, 0, 36]
        ret = self.arm.Movej_Cmd(joint=joints, v=speed, trajectory_connect=0)
        if ret == 0:
            return True
        else:
            return False
