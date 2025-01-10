import math
import os
import random
import threading
import time
from functools import wraps
from threading import Timer

import numpy as np
import yaml

try:
    from .agilex_api import HttpClient, WSClient, png_coordinate_to_map, GLOBAL_STATUS_DICT, GLOBAL_CONTROL_MODE_DICT, \
        quaternion_to_euler_dict, quaternion_to_euler
    from .point_cloud_process import obstacle_detect
    from .task_config import global_logger
    from .tools import smooth_acc, calculate_vector_angle, cal_distance
    from .QR_detect import QR_Detector
except:

    from agilex_api import HttpClient, WSClient, png_coordinate_to_map, GLOBAL_STATUS_DICT, GLOBAL_CONTROL_MODE_DICT, \
        quaternion_to_euler_dict, quaternion_to_euler
    from point_cloud_process import obstacle_detect
    from task_config import global_logger
    from tools import smooth_acc, calculate_vector_angle, cal_distance
    from QR_detect import QR_Detector

ws_url = "ws://192.168.1.102:9090"  # 填写实际的机器人IP地址
http_url = "http://192.168.1.102/apiUrl"
token = None


def singleton(cls):
    """Set class to singleton class.

    :param cls: class
    :return: instance
    """
    __instances__ = {}

    @wraps(cls)
    def get_instance(*args, **kw):
        """Get class instance and save it into glob list."""
        if cls not in __instances__:
            __instances__[cls] = cls(*args, **kw)
        return __instances__[cls]

    return get_instance


@singleton
class Robot():
    def __init__(self):

        ## 加载配置文件
        self._load_config()

        ## http客户端
        self.http_client = HttpClient(http_url)
        self.http_client.login_()

        ### websocket 客户
        self.ws_client = WSClient(ws_url)

        ## 发送心跳信号， 维持连接
        self.ws_client.heart_beat()

        ## 设置导航参数
        self.ws_client.set_navi_params(max_vel=self.config.get('move_control').get('max_vel'),
                                       max_acc=self.config.get('move_control').get('mav_acc'))

        ## 设置自动充电
        self.is_charge_set = False
        ## 订阅数据
        self.sub_status_and_data()

        ## 地图信息
        self.map_info = None

        ## 巡航点设置和管理
        self.cruise_points_index_cur = 0  # 当前巡航点索引
        # self.total_cruise_point_num = len(cruise_point_name_list)  # 总巡航点个数
        self.is_task_finished = False  ##当前任务是否完成， 完成则执行下一个任务
        self.is_task_running = False  ## 当前是否在执行任务中， 如果在执行任务中， 就不再监控机器人状态
        self.is_cruise_command_send = False  # 判断当前巡航点是否已下发执行

        ## 任务管理
        self.target_pose = None  ## 当前的任务目标点
        self.sub_target_pose = None  ##任务异常时会切分任务目标， 子任务目标点

        ## 状态监控管理
        self.robot_stop_status_list = []
        self.robot_location_status_list = []
        self.status_monitor()  ## 启动监控管理

        self.move_distance = 0  ## 机器人移动的距离
        self.rotate_angular = 0  ## 机器人转动的角度
        ##
        self.QR_detector = QR_Detector()

    def _load_config(self):
        # file_path = os.path.abspath(__file__)
        with open(os.path.join("/home/nvidia/leo/robot/robot_sdk", "config.yaml"), 'rb') as f:
            self.config = yaml.safe_load(f)
        self.map_name = self.config.get("task").get("map_name")

    def sub_status_and_data(self):
        ## 获取当前状态信息
        self.ws_client.sub_slam_status()

        self.ws_client.sub_nav_status()

        self.ws_client.sub_task_status()

        self.ws_client.sub_robot_satus()

        self.ws_client.sub_battery_satus()

        self.ws_client.sub_task_progress()

        self.ws_client.sub_robot_pos()

        self.ws_client.sub_plan_path()

        self.ws_client.sub_localiztion_status()

        self.ws_client.recieve_data()

        time.sleep(2)

    def get_robot_pose(self, need_angle=False):
        res = self.ws_client.get_robot_pos()
        if res is not None:
            pose = res.get("msg").get("pose").get("pose")
            timestamp = res.get("msg").get("header").get("stamp")
            # pose = res.get('msg').get("odom_pose").get('pose').get("position")
            x, y, z = pose.get("position").get("x"), pose.get("position").get("y"), pose.get("position").get("z")
            orientation = pose.get("orientation")
            pose = [x, y, z], orientation, timestamp

            if need_angle:
                angle = quaternion_to_euler_dict(ori=orientation)
                pose = [x, y, z], orientation, timestamp, angle

        else:
            pose = None

        return pose

    def get_task_status(self):
        res = self.ws_client.get_task_status()
        if res is not None:
            msg = res.get('msg')
            print(f"当前导航任务名称：{msg.get('task_name')}, 任务总循环次数： {msg.get('total_looptimes')}, "
                  f"当前循环到第{msg.get('loop_index')}次任务, 当前任务执行到第{msg.get('task_index')}个目标",
                  f"是否到达当前目标点:{msg.get('isGoalReach')}")

    def get_task_progress(self):
        res = self.ws_client.get_task_progress()
        progress = None
        if res is not None:
            progress = res.get('msg').get('data')
        return progress

    def get_global_status(self, to_str=False):
        """
        获取全局状态信息：
        导航系统的状态，可能存在的值
        0 : 等待新任务
        1 : 正在执行任务
        2 : 取消任务
        3 : 任务完成
        4 : 任务中断
        5 : 任务暂停
        6 : 定位丢失
        7 : 异常状态,一般最后一个点无法到达会反馈此状态
        8 : 任务异常暂停，一般中间某个点无法到达会反馈此状态
        9 : 正在充电
        10 返回充电桩
        11 : 正在出桩
        12 : 返回起点
        13 : 未出桩
        14 : 等待充电信号
        """
        global_status_data = self.ws_client.get_nav_status()
        if global_status_data is not None:
            status = global_status_data.get("msg").get("status")
            if to_str:
                status = GLOBAL_STATUS_DICT.get('status')
        else:
            status = None
        return status

    def get_nav_status(self, need_map_name=False):
        """
        获取导航系统的状态：
        op	object	回复"subscribe " " 固定值publish
        topic	object	回复对应的topic
        nav	object	导航状态信息
        mapping_3d	object	3D建图状态
        mapping_2d	object	2D建图状态
        record	object	录包状态
        state	bool	true-开启,false-关闭
        name	string	正在导航的地图名称,或者当前录制的地图包名称
        task	string	当前地图正在使用的任务名称
        """
        nav_status_data = self.ws_client.get_slam_status()
        res = None
        if nav_status_data is not None:
            nav_status = nav_status_data.get("msg").get("nav").get("state")
            if need_map_name:
                cur_map_name = nav_status_data.get("msg").get("nav").get("name")
                res = nav_status, cur_map_name
            else:
                res = nav_status
        return res

    def get_robot_status(self):
        """
        获取底盘数据：
        op	string	回复"subscribe "  固定值publish
        topic	string	回复对应的topic
        angular_velocity	float	当前角速度
        linear_velocity	float	当前线速度
        fault_code	int	底盘错误信息反馈
        base_state	int	当前底盘状态 :0-正常状态
        control_mode	int	当前底盘控制模式
        0-空闲待机状态
        1-CAN指令控制
        2-串口控制(不支持)
        3-手柄遥控控制
        """
        robot_status_data = self.ws_client.get_robot_status()

        if robot_status_data is not None:
            control_mode = robot_status_data.get('msg').get("control_mode")
            base_state = robot_status_data.get('msg').get("base_state")
            angular_velocity = robot_status_data.get('msg').get("angular_velocity")
            linear_velocity = robot_status_data.get('msg').get("linear_velocity")
            robot_status = [control_mode, angular_velocity, linear_velocity, base_state]
        else:
            robot_status = None

        return robot_status

    def get_nav_path(self):
        nav_path_data = self.ws_client.get_nav_path()
        pose_list = []
        if nav_path_data is not None:
            init_pose_list = nav_path_data.get("msg").get("poses")
            # global_logger.info(f"导航路径点个数： {len(init_pose_list)}")
            if len(init_pose_list) > 0:
                for init_pose in init_pose_list:
                    position = init_pose.get('pose').get('position')
                    orientation = init_pose.get('pose').get('orientation')
                    # global_logger.warning(f"路径信息：{init_pose}")
                    x, y = position.get('x'), position.get('y')
                    theta = quaternion_to_euler_dict(ori=orientation)

                    pose_list.append([x, y, theta])

        return pose_list

    def get_battery_info(self):
        return self.ws_client.get_battery_status()

    def is_goal_reach(self):
        # not used
        res = self.ws_client.get_task_status()
        # print("是否到达目标点结果：", res)
        # print("*" * 20, "is goal reach", res)
        flag = False
        if res is not None:
            flag = res.get('msg').get('isGoalReach')
        # print("10" * 20, flag)
        return flag

    def is_reached_target(self, target_pos=None, error_threshold=None):

        def _is_near_target(error_threshold):
            cur_pose = self.get_robot_pose()

            if cur_pose is None:
                return False

            pos = cur_pose[0]
            ori = cur_pose[1]
            yaw = quaternion_to_euler_dict(ori)

            pos_target = target_pos[:2]
            yaw_target = target_pos[2]

            distance = math.sqrt((pos_target[0] - pos[0]) ** 2 + (pos_target[1] - pos[1]) ** 2)
            angle_diff = abs(yaw_target - yaw)
            if angle_diff > 180:  ## 跨越正负180
                angle_diff = 360 - angle_diff

            # global_logger.warning(
            #     f"时间：{time.time()}目标位置：{target_pos}， 当前位置： {pos}, {yaw}, distance_diff: {distance}， angle_diff: {angle_diff}")
            if distance <= error_threshold[0] and angle_diff <= error_threshold[1]:
                return True
            else:
                return False

        def _system_reached():
            retry_time = 10
            for i in range(retry_time):
                status = self.get_global_status()
                if status == 3:
                    return True
                time.sleep(0.1)
            return False

        ## 容忍误差为0时， 采用系统的判断方式
        if error_threshold is None:
            return _system_reached()

        ## 当有容忍误差时， 采用自定义判断方式
        return _is_near_target(error_threshold=error_threshold)

    # def create_map(self, map_name="office"):
    #     time_1 = 5
    #     time_2 = 5
    #
    #     ## 录制bag数据包
    #     self.ws_client.record_bag(optype="start", filename=map_name)
    #     time.sleep(time_1)
    #     self.ws_client.record_bag(optype="stop", filename=map_name)
    #     time.sleep(time_2)
    #
    #     ## 进行3D建图
    #     self.ws_client.mapping_3d(optype="start", filename=map_name)
    #     time.sleep(time_1)
    #     self.ws_client.mapping_3d(optype="stop", filename=map_name)
    #     time.sleep(time_2)
    #
    #     ## 进行2D建图
    #     self.ws_client.mapping_2d(optype="start", filename=map_name)
    #     time.sleep(time_1)
    #     self.ws_client.mapping_2d(optype="stop", filename=map_name)
    #     time.sleep(time_2)
    #
    #     # ###获取地图列表
    #     self.http_client.get_maplist()
    #     print(f"地图列表：{self.http_client.map_list}")

    # def run_cruise_task(self):
    #     # 给一个巡航点， 执行一次移动到巡航点任务
    #     # 由于是在外部定时器中执行， 所以需要判断是否已经发过巡航指令， 避免重复发送
    #     if not self.is_cruise_command_send:
    #         self.is_cruise_command_send = True
    #
    #         target_pose = cruise_points_dict.get(cruise_point_name_list[self.cruise_points_index_cur])
    #         global_logger.info(f"准备执行巡航任务， 巡航点名称：{cruise_point_name_list[self.cruise_points_index_cur]}")
    #         self.move_to_target(target_pos=target_pose)
    #
    #     ## 更新巡航点
    #     if self.is_task_finished:
    #         global_logger.info(
    #             f"当前巡航点： {cruise_point_name_list[self.cruise_points_index_cur]}完成")
    #         self.cruise_points_index_cur += 1
    #         self.cruise_points_index_cur = self.cruise_points_index_cur % self.total_cruise_point_num
    #         global_logger.info(f"下一个巡航点： {cruise_point_name_list[self.cruise_points_index_cur]} 即将开始")
    #         self.is_cruise_command_send = False  # 状态复位
    #         self.is_task_finished = False  # 状态复位
    #         self.is_task_running = False

    def _coord_transform(self, png_coord):
        if self.map_info is not None:
            map_x, map_y = png_coordinate_to_map(pos=png_coord, map_info=self.map_info)
            return [map_x, map_y, png_coord[2]]
        else:

            print(f"地图信息为空， 无法进行坐标转换， 请检查！！")

    def init_pose(self, pos):

        retry_time = 30
        for i in range(retry_time):
            self.ws_client.initial_pos(pos=pos)
            time.sleep(0.2)
            status = self.get_global_status()
            if status != 6 and not self.ws_client.get_localization_lost():
                return True
        return False

    def change_map(self, new_map_name):
        self.ws_client.change_map(filename=new_map_name)
        _, cur_map_name = self.get_nav_status(need_map_name=True)
        max_try_time = 10
       # for i in range(max_try_time):
        #    if cur_map_name == new_map_name:
         #       return True
          #  time.sleep(0.05)
           # self.ws_client.change_map(filename=new_map_name)
        return True

    def check_robot_status(self, map_name=None):
        ##检查导航是否开启， 如果未开启， 则开启导航
        if map_name is None:
            map_name = self.map_name
        nav_status = self.get_nav_status()
        global_logger.info(f"开启导航, 地图：{map_name}")
        self.ws_client.navigation(optype="start", filename=map_name)
        time.sleep(0.5)
        global_logger.info(f"导航是否开启： {nav_status}")
        if not nav_status:
            global_logger.warning(f"导航未开启,将开启导航！！！")
            self.ws_client.navigation(optype="start", filename=map_name)
            time.sleep(0.5)
            nav_status = self.get_nav_status()
            global_logger.info(f"导航是否开启： {nav_status}")

        ##设置自动充电功能
        ##由于设置自动充电的前提是导航状态已开启， 因此在这里设置， 并且只需设置一次
        if not self.is_charge_set:
            self.is_charge_set = True
            self.ws_client.set_charge_level(threshold=[self.config.get('charge').get('min_charge_level'),
                                                       self.config.get('charge').get('max_charge_level')])

        ## 检查全局任务状态， 如果正在执行任务， 则取消任务重新开始
        status = self.get_global_status()
        global_logger.info(f"导航系统当前状态： {GLOBAL_STATUS_DICT.get(status)}")
        if status is None:
            global_logger.warning(f"未获取到导航系统状态！！！")
        elif status == 6:
            global_logger.warning(f"定位丢失！！！")

        elif status == 1:
            global_logger.warning(f"当前有正在进行中的任务， 将结束旧任务， 重新开始新任务！！！")
            self.ws_client.cancel_nav()
            time.sleep(0.5)
            status = self.get_global_status()
            global_logger.info(f"导航系统当前状态： {GLOBAL_STATUS_DICT.get(status)}")

        ## 检查底盘控制模式
        ## 由于手柄模式无法自动移动， 因此用while True进行阻塞
        # while True:
        #     robot_status = self.get_robot_status()
        #     if robot_status is not None:
        #         control_mode = robot_status[0]
        #         # global_logger.info(f"当前控制模式：{GLOBAL_CONTROL_MODE_DICT.get(control_mode)}")
        #         if control_mode != 3:  # 3: "手柄遥控控制"
        #             break
        #         else:
        #             global_logger.warning(f"当前处于遥控模式， 无法自动运行。")
        #             time.sleep(0.5)
        #     else:
        #         global_logger.info("未获取到控制模式数据！")
        #         time.sleep(0.5)

        robot_status = self.get_robot_status()
        if robot_status is not None:
            control_mode = robot_status[0]
            global_logger.info(f"当前控制模式：{GLOBAL_CONTROL_MODE_DICT.get(control_mode)}")

    def move_to_target(self, target_pos=[0, 0, 0], update_target_pose=True, map_name=None):
        """控制机器人移动到目标位置。机器人可能因为异常无法移动到目标位置。"""

        if map_name is None:
            map_name = self.map_name

        if update_target_pose:
            self.target_pose = target_pos  ## 异常处理时目标点位信息不更新， 以便异常处理后再次执行目标点

        ##输入要求是一个list,包含x, y， angle 三个值， 如果不是需要进行转换
        if not isinstance(target_pos, list):
            target_pos = [target_pos.x, target_pos.y, target_pos.z]

        global_logger.info(f"机器人准备移动到位置：{target_pos}")

        global_logger.info(f"开始检查机器人当前状态...")
        self.check_robot_status(map_name=map_name)
        global_logger.info(f"结束检查机器人当前状态")

        ## 获取地图信息
        self.map_info = self.http_client.get_map_info(map_name)
        global_logger.info(f"当前使用的地图信息： {self.map_info}")

        ## 下发机器人移动指令
        global_logger.info(f"开始下发机器人移动指令...")
        for i in range(3):
            # status = self.get_global_status()
            sucess_flag = self.http_client.run_realtime_task(pos=target_pos)  ##执行导航
            if not sucess_flag:
                self.http_client.run_realtime_task(pos=target_pos)  ##执行导航
            time.sleep(0.05)
        global_logger.info(f"结束下发机器人移动指令")

    def naive_move(self, goal_distance=0, goal_angular=0, vel_linear_x=0.3, vel_linear_y=0, vel_angular=1):

        # assert (vel_linear_x == 0 or vel_linear_y == 0)

        rate = 10
        # vel_linear = 0.3  # m/s
        # vel_angular = 1  # rad/s
        # if goal_distance < 0:
        #     vel_linear = -1 * vel_linear
        # if goal_angular < 0:
        #     vel_angular = -1 * vel_angular

        ## 先转动角度， 然后直线移动
        angular_duration = goal_angular / 57.3 / vel_angular
        angular_ticks = int(abs(rate * angular_duration))

        vel_linear = np.sqrt(np.square(vel_linear_x) + np.square(vel_linear_y))
        linear_duration = goal_distance / vel_linear
        linear_ticks = int(abs(rate * linear_duration))

        self.ws_client.navigation(optype="start", filename=self.map_name)

        if goal_angular != 0:
            global_logger.warning(f"无导航下转动底盘， 目标角度：{goal_angular}")
            for i in range(angular_ticks):
                self.ws_client.naive_move(vel_angular=vel_angular)
                time.sleep(1 / rate)

        if goal_distance != 0:
            global_logger.warning(f"无导航下移动底盘， 目标距离：{goal_distance}")
            for i in range(linear_ticks):
                self.ws_client.naive_move(vel_linear_x=vel_linear_x, vel_linear_y=vel_linear_y)
                time.sleep(1 / rate)

    def naive_move_with_obstacle_avoid(self, goal_distance=0, goal_angular=0):

        safe_distance = 0.3  # m

        rate = 10
        vel_linear = 0.3  # m/s
        vel_angular = 1  # rad/s
        vel_angular = 1  # rad/s
        if goal_distance < 0:
            vel_linear = -1 * vel_linear
        if goal_angular < 0:
            vel_angular = -1 * vel_angular

        ## 先转动角度， 然后直线移动
        angular_duration = goal_angular / 57.3 / vel_angular
        angular_ticks = int(abs(rate * angular_duration))

        linear_duration = goal_distance / vel_linear
        linear_ticks = int(abs(rate * linear_duration))

        self.ws_client.navigation(optype="start", filename=self.map_name)

        global_logger.warning(f"无导航下转动底盘， 目标角度：{goal_angular}")

        idx = 0
        while idx <= angular_ticks:
            point_cloud = self.ws_client.get_pointCloud2_compress()
            obstacle_dis = obstacle_detect(point_cloud_data=point_cloud)
            if obstacle_dis is not None and obstacle_dis >= safe_distance:
                self.ws_client.naive_move(vel_linear=0, vel_angular=vel_angular)
                time.sleep(1 / rate)
                idx += 1

        global_logger.warning(f"无导航下移动底盘， 目标距离：{goal_distance}")
        idx = 0
        while idx <= linear_ticks:
            point_cloud = self.ws_client.get_pointCloud2_compress()
            obstacle_dis = obstacle_detect(point_cloud_data=point_cloud)
            if obstacle_dis is not None and obstacle_dis >= safe_distance:
                self.ws_client.naive_move(vel_linear=vel_linear, vel_angular=0)
                time.sleep(1 / rate)

    def status_monitor(self):

        robot_pose = self.get_robot_pose()
        # global_logger.warning(f"robot_pose: {robot_pose}")
        ###定位异常， 机器人停止等异常的检测需要累计一段时间，因此需要定时进行检测， 累计一段时间的状态进行判断

        num = int(self.config.get("monitor").get("exception_interval") / self.config.get("monitor").get("period"))
        self.robot_location_status_list = self.robot_location_status_list[-1 * num:]
        self.robot_stop_status_list = self.robot_stop_status_list[-1 * num:]

        location_status = self.get_global_status()
        if location_status == 6 and not self.is_task_running:  ##任务在进行中， 就不计入异常
            self.robot_location_status_list.append(1)
        else:
            self.robot_location_status_list.append(0)

        velocity = self.get_robot_status()
        global_logger.debug(f"机器人当前速度信息：{velocity}")
        if velocity is not None and abs(velocity[1]) < self.config.get("monitor").get(
                "min_velocity_linear") and abs(
            velocity[2]) < self.config.get("monitor").get(
            "min_velocity_angle") and not self.is_task_running:  ##任务在进行中， 就不计入异常
            self.robot_stop_status_list.append(1)
        else:
            self.robot_stop_status_list.append(0)
        global_logger.debug(f"stop status:{self.robot_stop_status_list}")

        timer = Timer(interval=self.config.get("monitor").get("period"), function=self.status_monitor)
        timer.start()  ##启动定时监控

    def is_robot_stop(self):
        ### 累计一段时间的平均速度，小于一定阈值则视为机器人停止
        num = int(self.config.get("monitor").get("exception_interval") / self.config.get("monitor").get("period"))
        if sum(self.robot_stop_status_list) >= num:
            return True
        else:
            return False

    def is_robot_location_error(self):
        num = int(self.config.get("monitor").get("exception_interval") / self.config.get("monitor").get("period"))
        if sum(self.robot_location_status_list) >= num:
            return True
        else:
            return False

    def is_not_reachable(self):
        status = self.get_global_status()
        if status in [7, 8]:
            return True
        else:
            return False

    def _move_random(self):
        cur_pose = self.get_robot_pose()
        if cur_pose is not None:
            x, y, z = cur_pose[0]
            angle = quaternion_to_euler_dict(cur_pose[1])

            delta_x = random.uniform(0, 0.2)
            delta_y = random.uniform(-0.1, 0.1)
            delta_angle = random.uniform(-90, 90)

            next_target_pose = [x + delta_x, y + delta_y, angle + delta_angle]
            self.sub_target_pose = next_target_pose  ##设定子任务目标点
            global_logger.warning(
                f"机器人将随机移动以校正错误。当前位置： {[x, y, angle]}, 下一个目标点：{next_target_pose}")
            self.move_to_target(target_pos=next_target_pose, update_target_pose=False)  # 不要更新最终目标点位信息

    def location_error_process(self):
        ## 如果定位丢失， 则在当前位置的附近随机探索， 让机器人在移动中逐渐找到定位
        self._move_random()

    def not_reachable_process(self):

        self._move_random()
        # nav_pose_list = self.get_nav_path()
        # if nav_pose_list is not None:
        #     last_pose = nav_pose_list[-1].get("pose")
        #     position = last_pose.get('position')
        #     orientation = last_pose.get('orientation')
        #     x, y = position.get('x'), position.get('y')
        #     angle = quaternion_to_euler_dict(orientation)
        #     next_target_pose = [x, y, angle]
        #     self.sub_target_pose = next_target_pose  ##设定子任务目标点
        #     global_logger.warning(f"机器人无法到达目标位置，将对任务进行切分。下一个目标点： {next_target_pose}")
        #     self.move_to_target(target_pos=next_target_pose, update_target_pose=False)

    def stop_error_process(self):
        ##如果检测到机器人异常停止， 则随机移动机器人
        self._move_random()

    def process_exception(self):
        """
        主要处理3个异常：
        1 机器人定位异常
        2 目标位置无法到达
        3 机器人异常停下
        """

        if self.is_robot_location_error():
            global_logger.warning(f"定位异常")
            self.location_error_process()


        elif self.is_not_reachable():
            global_logger.warning(f"目标点无法到达")
            self.not_reachable_process()

        elif self.is_robot_stop():
            global_logger.warning(f"机器人异常停止运动")
            self.stop_error_process()

        ## 如果发生异常，且子任务目标完成， 则继续执行最终目标点任务
        if self.sub_target_pose is not None:
            if self.is_reached_target(target_pos=self.sub_target_pose):  # 子任务完成
                self.sub_target_pose = None  # 删除子任务
                global_logger.info(f"子任务目标点完成， 异常解除。 将继续执行原目标任务。 目标点： {self.target_pose}")
                self.move_to_target(target_pos=self.target_pose)

    def pause_nav(self):
        self.ws_client.pause_nav()

    def continue_nav(self):
        self.ws_client.continue_nav()

    ## 外部更新机器人状态接口， 任务状态改变时通知
    def update_robot_state(self, **kwargs):
        if "is_task_finished" in kwargs:
            self.is_task_finished = kwargs.get('is_task_finished')
            global_logger.info(f"当前任务点完成， is_task_finished参数更新， 当前状态： {self.is_task_finished}")
        if "is_task_running" in kwargs:
            self.is_task_running = kwargs.get('is_task_running')
            global_logger.info(f"当前正任务正在运行中， is_task_running参数更新， 当前状态： {self.is_task_running}")

    def get_point_cloud(self):
        data = self.ws_client.get_pointCloud2()
        point_cloud = None
        if data is not None:
            point_cloud = list(map(generate_xyz, data))
            point_cloud = np.array(point_cloud)
            print(f"点云大小： {point_cloud.shape}")
        return point_cloud

    def imu_monitor(self, stop_event):
        self.ws_client.sub_imu_data()  ## 订阅imu数据

        v0_x, v0_y = 0, 0
        s0_x, s0_y = 0, 0

        last_angular = None
        total_angular = 0
        # acc_offset_norm = 9.827884938199292

        acc_x_list = []
        acc_y_list = []
        time_last = None

        acc_offset_x = None
        acc_offset_y = None

        # init_angular = None

        while not stop_event.is_set():
            imu_data = self.ws_client.get_imu_data()

            if imu_data is not None:
                ## 计算距离
                time_now = time.time()
                if time_last is None:
                    delta_t = 0
                else:
                    delta_t = time_now - time_last
                time_last = time_now
                acc = imu_data.get("linear_acceleration")
                acc_x, acc_y, acc_z = acc.get("x"), acc.get("y"), acc.get("z")

                acc_x_list.append(acc_x)
                acc_y_list.append(acc_y)

                # acc_norm = np.sqrt(np.square(acc_x) + np.square(acc_y) + np.square(acc_z))
                # acc_norm = np.sqrt(np.square(acc_x) + np.square(acc_y))

                # acc_norm_list.append(acc_norm)

                sample_point = 60  # 至少60个采样点， 大约需要累计0.3s
                # acc_norm_list = acc_norm_list[-1 * sample_point:]
                acc_x_list = acc_x_list[-1 * sample_point:]
                acc_y_list = acc_y_list[-1 * sample_point:]
                if len(acc_x_list) < sample_point:
                    continue  ##没有累计到足够imu数据， 则等待

                if acc_offset_x is None:
                    acc_offset_x = np.mean(acc_x_list)
                if acc_offset_y is None:
                    acc_offset_y = np.mean(acc_y_list)

                acc_x_smooth = smooth_acc(acc_x_list - acc_offset_x)[-1]
                acc_y_smooth = smooth_acc(acc_y_list - acc_offset_y)[-1]

                # acc = acc_norm - acc_offset_norm  ## 减去静止时的偏移

                v_x = v0_x + acc_x_smooth * delta_t
                v_y = v0_y + acc_y_smooth * delta_t
                s_x = s0_x + (v0_x + v_x) / 2 * delta_t
                s_y = s0_y + (v0_y + v_y) / 2 * delta_t

                v0_x, v0_y = v_x, v_y
                s0_x, s0_y = s_x, s_y

                self.move_distance = np.sqrt(np.square(s_x) + np.square(s_y))

                ## 计算角度
                cur_angular = quaternion_to_euler_dict(ori=imu_data.get("orientation"))
                #
                # if init_angular is None:
                #     init_angular = cur_angular

                # total_angular = cur_angular - init_angular
                #
                # if 90 < init_angular <= 180 and -180 <= cur_angular <= -90:
                #     total_angular +=360
                # elif -180<=init_angular<=-90 and
                #     total_angular = cur_angular - init_angular

                if last_angular is None:
                    last_angular = cur_angular
                cur_rotate_angular = cur_angular - last_angular

                ##处理正负180°附近的跳变, 顺时针转动角度为负， 逆时针转动角度为正
                if 90 < cur_angular <= 180 and -180 <= last_angular < -90:
                    cur_rotate_angular = cur_rotate_angular - 360
                elif -180 <= cur_angular < -90 and 90 <= last_angular <= 180:
                    cur_rotate_angular = cur_rotate_angular + 360

                total_angular += cur_rotate_angular
                #
                self.rotate_angular = total_angular
                # print(f"cur_angular: {cur_angular}, last_angular:{last_angular}")

                last_angular = cur_angular

                tmp = self.get_robot_status()
                if tmp is not None:
                    vel = tmp[2]
                else:
                    vel = None

                print(
                    f"移动距离： {self.move_distance} | s_x: {s_x} | s_y:{s_y}, 当前速度：{v_x}| {v_y}, 底盘速度：{vel}, 当前转动角度： {cur_rotate_angular}, 总转动角度：{self.rotate_angular} ")

            time.sleep(0.004)  ## 250Hz

        self.ws_client.unsub_imu_data()

    def naive_move_with_imu(self, goal_distance=0, goal_angular=0, vel_linear=0.3, vel_angular=1):

        rate = 20
        stop_event = threading.Event()
        imu_monitor_thread = threading.Thread(target=self.imu_monitor, args=(stop_event,))

        imu_monitor_thread.start()

        ## 先转动角度， 然后直线移动
        angular_duration = goal_angular / 57.3 / vel_angular
        angular_ticks = int(abs(rate * angular_duration))

        linear_duration = goal_distance / vel_linear
        linear_ticks = int(abs(rate * linear_duration))

        self.ws_client.navigation(optype="start", filename=self.map_name)

        time.sleep(0.5)  ## 等待0.5s， 累积imu数据

        if goal_angular != 0:
            global_logger.warning(f"无导航下转动底盘， 目标角度：{goal_angular}")
            while self.rotate_angular <= goal_angular:
                self.ws_client.naive_move(vel_linear=0, vel_angular=vel_angular)
                time.sleep(1 / rate)

        if goal_distance != 0:
            global_logger.warning(f"无导航下移动底盘， 目标距离：{goal_distance}")
            while self.move_distance <= goal_distance:
                self.ws_client.naive_move(vel_linear=vel_linear, vel_angular=0)
                time.sleep(1 / rate)

        stop_event.set()  ## 结束子线程

    def modify_yaw_with_QR(self, goal_yaw):

        ##根据二维码定位结果调整旋转角度

        def _calculate_vel_angular(diff_yaw=0):
            ## 速度控制， 角度差距大时速度快
            expected_time = 1.5
            init_vel_angular = abs(diff_yaw) / 57.3 / expected_time

            init_vel_angular = max(0.1, init_vel_angular)
            init_vel_angular = min(1, init_vel_angular)  ## 速度限制在0.1~1之间

            ##底盘逆时针旋转， 角度变大， 顺时针旋转， 角度变小
            if diff_yaw > 0:
                vel_angular = init_vel_angular  ## 逆时针转
            elif diff_yaw < 0:
                vel_angular = -1 * init_vel_angular  ##顺时针转

            if abs(diff_yaw) > 180:  ## 需要转动的角度大于180， 则反向旋转
                diff_yaw = 360 - abs(diff_yaw)
                vel_angular = -1 * vel_angular

            return diff_yaw, vel_angular

        def _calculate_diff_yaw():
            cur_pose = self.QR_detector.detect_pose_by_QR_code()

            cur_yaw = cur_pose.get('yaw')
            if -180 < cur_yaw < 0:
                cur_yaw += 360

            diff_yaw = goal_yaw - cur_yaw

            return cur_yaw, diff_yaw

        angular_diff_threshold = 3

        change_index = 0
        max_modify_time = 5
        cur_yaw, diff_yaw = _calculate_diff_yaw()

        global_logger.warning(f"当前角度：{cur_yaw}, 目标角度：{goal_yaw}, 差异：{diff_yaw}")

        while abs(diff_yaw) > angular_diff_threshold and change_index < max_modify_time:
            diff_yaw, vel_angular = _calculate_vel_angular(diff_yaw=diff_yaw)
            self.naive_move(goal_angular=diff_yaw, vel_angular=vel_angular)
            global_logger.warning(f"第{change_index}次角度调整完成, 检测当前位置...")

            time.sleep(0.5)  ## 必须要等待静止时测量， 否则测量误差很大
            _, diff_yaw = _calculate_diff_yaw()

            global_logger.warning(f"第{change_index}次角度调整完成, 距离目标角度还差{diff_yaw}度")
            change_index += 1
        global_logger.warning(f"角度调整完成")

    def modify_position_with_QR_fast(self, goal_x=0, goal_y=0, init_vel_linear=0.3):
        ## 位置调整分2步， 先前后平移， 然后横向移动， 需要计算出在2个方向上的移动分量
        def _calculate_xy():
            cur_pose = self.QR_detector.detect_pose_by_QR_code()
            cur_x, cur_y, cur_yaw = cur_pose.get('x'), cur_pose.get('y'), cur_pose.get('yaw')
            return [cur_x, cur_y], cur_yaw

        pose1, cur_yaw = _calculate_xy()

        # 当前方向与x轴之间夹角
        theta_x = cur_yaw - 90  ## yaw角的0 方向是y轴负向
        if theta_x < -180:
            theta_x += 360  ## 把角度映射到0~180 和-180~0 的范围
        # 通过单位向量计算坐标
        fake_x, fake_y = np.cos(theta_x / 57.3), np.sin(theta_x / 57.3)
        fake_x += pose1[0]
        fake_y += pose1[1]

        ##先求出当前位姿和目标位姿之间的夹角和旋转方向
        vector_angle, vertical_ori = calculate_vector_angle(vector1=[pose1, [fake_x, fake_y]],
                                                            vector2=[pose1, [goal_x, goal_y]])

        global_logger.warning(f"向量夹角：{vector_angle}, 方向： {vertical_ori}")

        distance_to_goal_now = cal_distance(point1=pose1, point2=[goal_x, goal_y])
        parallel_ori = 1
        if vector_angle > 90:
            parallel_ori = -1
        #
        # # 分垂直和水平两个方向分别移动
        # distance_vertical = abs(distance_to_goal_now * np.sin(vector_angle / 57.3))  ## 垂直移动距离
        # distance_parallel = abs(distance_to_goal_now * np.cos(vector_angle / 57.3))  ## 平移距离
        # self.naive_move(goal_distance=distance_vertical, vel_linear_y=init_vel_linear * vertical_ori, vel_linear_x=0)
        #
        # self.naive_move(goal_distance=distance_parallel, vel_linear_x=init_vel_linear * parallel_ori, vel_linear_y=0)

        ## 直接斜方向一次移动
        vel_linear_vertical = abs(init_vel_linear * np.sin(vector_angle / 57.3))
        vel_linear_parallel = abs(init_vel_linear * np.cos(vector_angle / 57.3))
        global_logger.warning(
            f"斜方向移动，vel_x: {vel_linear_parallel * parallel_ori}, vel_y:{vel_linear_vertical * vertical_ori}")
        self.naive_move(goal_distance=distance_to_goal_now, vel_linear_x=vel_linear_parallel * parallel_ori,
                        vel_linear_y=vel_linear_vertical * vertical_ori)

    def modify_position_with_QR(self, goal_x=0, goal_y=0, attempt_distance=0.05, attempt_vel=0.1, init_vel_linear=0.3):
        ## 位置调整分2步， 先前后平移， 然后横向移动， 需要计算出在2个方向上的移动分量
        # 直接计算出移动方向较为困难， 采用试探的方法， 先朝一个方向移动， 如果离目标位置变远， 说明方向反了

        def _calculate_xy():
            cur_pose = self.QR_detector.detect_pose_by_QR_code()
            cur_x, cur_y = cur_pose.get('x'), cur_pose.get('y')
            return [cur_x, cur_y]

        time.sleep(0.5)
        pose1 = _calculate_xy()
        distance_to_goal = cal_distance(point1=pose1, point2=[goal_x, goal_y])

        # ## 先移动10cm, 找出一个点， 求方向

        global_logger.warning("水平移动...")
        self.naive_move(goal_distance=attempt_distance, vel_linear_x=attempt_vel)

        time.sleep(0.5)
        pose2 = _calculate_xy()
        vector_angle = calculate_vector_angle(vector1=[pose1, pose2],
                                              vector2=[pose2, [goal_x, goal_y]])

        global_logger.warning(f"向量夹角： {vector_angle}")

        distance_to_goal_now = cal_distance(point1=pose2, point2=[goal_x, goal_y])

        distance_vertical_init = abs(distance_to_goal_now * np.sin(vector_angle / 57.3))  ## 垂直移动距离
        distance_parallel = abs(distance_to_goal_now * np.cos(vector_angle / 57.3))  ## 平移距离
        print(f"垂直距离：{distance_vertical_init}, 水平距离： {distance_parallel}")

        vel_linear = init_vel_linear

        if vector_angle > 90:
            vel_linear = -1 * vel_linear

        self.naive_move(goal_distance=distance_parallel, vel_linear_x=vel_linear)  ## 平移

        ## 垂直移动
        print("垂直移动...")
        vel_linear = init_vel_linear
        self.naive_move(goal_distance=attempt_distance, vel_linear_x=0, vel_linear_y=attempt_vel)  ##试探性移动
        time.sleep(0.5)
        pose3 = _calculate_xy()
        distance_to_goal_now = cal_distance(point1=pose3, point2=[goal_x, goal_y])
        print(f"垂直距离: {distance_to_goal_now}")

        if distance_to_goal_now > distance_vertical_init:  ## 移动方向反了
            vel_linear = vel_linear * -1

        self.naive_move(goal_distance=distance_to_goal_now, vel_linear_x=0, vel_linear_y=vel_linear)  ## 垂直移动

    def naive_move_with_QR(self,
                           goal_pose={'x': 0.2798430391972063, 'y': -0.005058378594681704, 'yaw': 143.46474063718446}):

        goal_x, goal_y, goal_yaw = goal_pose.get('x'), goal_pose.get('y'), goal_pose.get('yaw')
        self.ws_client.navigation(optype="start", filename=self.map_name)

        global_logger.warning(f"开启相机流")
        ### 开启相机流
        self.QR_detector.start_camera_stream()
        global_logger.warning(f"开启相机流成功")

        ###############################角度移动########################################

        self.modify_yaw_with_QR(goal_yaw=goal_yaw)

        #########################################位置移动########################################
        time.sleep(0.5)
        cur_pose = self.QR_detector.detect_pose_by_QR_code()

        distance_diff = cal_distance(point1=[cur_pose.get('x'), cur_pose.get('y')], point2=[goal_x, goal_y])

        ## 迭代调整
        change_index = 0
        attempt_distance = 0.05  ## 距离太小， 调整完后的方向判断不准确， 至少5cm
        attempt_vel = 0.1
        init_vel_linear = 0.2

        max_modify_time = 5

        while abs(distance_diff) > 0.03 and change_index < max_modify_time:
            self.modify_position_with_QR_fast(goal_x=goal_x, goal_y=goal_y, init_vel_linear=init_vel_linear)

            attempt_distance *= 0.5
            attempt_vel *= 0.5
            init_vel_linear *= 0.5

            global_logger.warning(f"第{change_index}次位置调整完成, 检测当前位置...")
            change_index += 1
            time.sleep(0.5)  ## 必须要等待静止时测量， 否则测量误差很大
            cur_pose = self.QR_detector.detect_pose_by_QR_code()

            distance_diff = cal_distance(point1=[cur_pose.get('x'), cur_pose.get('y')], point2=[goal_x, goal_y])
            global_logger.warning(f"第{change_index}次位置调整完成, 位置误差：{distance_diff}.")

            self.modify_yaw_with_QR(goal_yaw=goal_yaw)

        global_logger.warning(f"位置调整完成")
        self.QR_detector.stop_camera_stream()  ## 关闭相机流
        global_logger.warning(f"关闭相机流")

        # cv2.destroyAllWindows()

        return True


def naive_move_test():
    robot = Robot()
    print(f"开启导航...")

    time_start = time.time()
    robot.ws_client.navigation(optype="start", filename="room_306")
    print(f"直接移动机器人...")

    robot.naive_move(goal_distance=1.5, goal_angular=0)

    print('*' * 100)
    print(f"移动耗时： {time.time() - time_start}")


import struct


def uint8_to_float32(uint8_list):
    # 将uint8转换为一个字节序列
    b1, b2, b3, b4 = uint8_list
    bytes = struct.pack('<BBBB', b1, b2, b3, b4)  ## 数据是小端存储
    # 使用unpack解包为float32
    return struct.unpack('f', bytes)[0]


def generate_xyz(xyz_uint):
    x, y, z = uint8_to_float32(xyz_uint[:4]), uint8_to_float32(xyz_uint[4:8]), uint8_to_float32(xyz_uint[8:12])
    return [x, y, z]


# def point_cloud_test():
#     robot = Robot()
#     print(f"开启导航...")
#     robot.ws_client.navigation(optype="start", filename="room_306")
#     print(f"订阅激光雷达点云数据...")
#     robot.ws_client.sub_pointCloud2_compress()
#
#     for scan_id in range(1000):
#         data = robot.ws_client.get_pointCloud2_compress()
#
#         if data is not None:
#             point_cloud = list(map(generate_xyz, data))
#             point_cloud = np.array(point_cloud)
#             print(f"点云大小： {point_cloud.shape}")
#             # print(point_cloud[:5])
#
#             time_start = time.time()
#             obstacle_dis = obstacle_detect(point_cloud_data=point_cloud)
#             print(f"处理耗时： {time.time() - time_start}")
#             # print(f"最近障碍物距离：{obstacle_dis}")
#             # save_point = np.zeros(shape=(data.shape[0], 3))
#             # init_x, init_y, init_z = data[:, :4], data[:, 4:8], data[:, 8:12]
#             # for i in range(len(save_point)):
#             #     x, y, z = uint8_to_float32(init_x[i]), uint8_to_float32(init_y[i]), uint8_to_float32(init_z[i])
#             #     save_point[i][0] = x
#             #     save_point[i][1] = y
#             #     save_point[i][2] = z
#             #
#             # print("*" * 20, save_point.shape)
#             #
#             # np.save("point_cloud.npy", save_point)
#             # write_ply_file_by_plyfile(point_list=point_cloud, point_cloud_file=f"point_cloud_{scan_id}.ply")
#
#         time.sleep(0.1)

def point_cloud_test():
    robot = Robot()
    print(f"开启导航...")
    robot.ws_client.navigation(optype="start", filename="room_306")
    print(f"订阅激光雷达点云数据...")
    robot.ws_client.sub_pointCloud2_compress()

    for scan_id in range(10000000):
        data = robot.ws_client.get_pointCloud2_compress()
        time_start = time.time()
        obstacle_dis = obstacle_detect(data)
        print(f"点云处理耗时：{time.time() - time_start}")

        # if data is not None:
        #     point_cloud = list(map(generate_xyz, data))
        #     point_cloud = np.array(point_cloud)
        #     print(f"点云大小： {point_cloud.shape}")
        #     # print(point_cloud[:5])
        #
        #     time_start = time.time()
        #     obstacle_dis = obstacle_detect(point_cloud_data=point_cloud)
        #     print(f"处理耗时： {time.time() - time_start}")
        #     # print(f"最近障碍物距离：{obstacle_dis}")
        # save_point = np.zeros(shape=(data.shape[0], 3))
        # init_x, init_y, init_z = data[:, :4], data[:, 4:8], data[:, 8:12]
        # for i in range(len(save_point)):
        #     x, y, z = uint8_to_float32(init_x[i]), uint8_to_float32(init_y[i]), uint8_to_float32(init_z[i])
        #     save_point[i][0] = x
        #     save_point[i][1] = y
        #     save_point[i][2] = z
        #
        # print("*" * 20, save_point.shape)
        #
        # np.save("point_cloud.npy", save_point)
        # write_ply_file_by_plyfile(point_list=point_cloud, point_cloud_file=f"point_cloud_{scan_id}.ply")

        time.sleep(0.1)


def save_point_cloud(data, save_name='point_cloud.ply'):
    import plyfile

    face = np.array([0, 1, 2])

    # 保存为PLY文件
    ply = plyfile.PlyElement.describe(np.array(data), 'vertex', comments=['vertices'])
    ply['face'] = plyfile.PlyElement(face, 'face')
    ply.write(save_name)
    print(f"点云文件已保存：{save_name}")


def write_ply_file_by_plyfile(point_list, point_cloud_file):
    import plyfile
    print("Writting PLY file".center(120, "="))
    points = [(point_list[i, 0], point_list[i, 1], point_list[i, 2]) for i in range(point_list.shape[0])]
    points = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    color = [(255, 255, 255)] * point_list.shape[0]
    color = np.array(color, dtype=[('red', 'uint8'), ('green', 'uint8'), ('blue', 'uint8')])

    # 定义PLY元素格式：这里我们只定义了一个名为vertex的元素，包含x、y、z属性
    vertex_element = plyfile.PlyElement.describe(points, name='vertex', comments=['x', 'y', 'z'])
    color = plyfile.PlyElement.describe(color, name="color", comments=['red', 'green', 'blue'])

    # 创建PlyData对象并写入PLY文件
    # 设置text=False并指定字节序为little_endian， 使用二进制格式会是的文件存储更小，但是里面的信息就不是文本了。如果不想要二进制存储，只需要将text设置为True就好了。
    ply_data = plyfile.PlyData([vertex_element, color], text=True, byte_order='<')
    ply_data.write(point_cloud_file)
    print(f"点云{point_cloud_file}存储完成!\n")


def imu_test():
    robot = Robot()
    print(f"开启导航...")

    time_start = time.time()
    # robot.ws_client.navigation(optype="start", filename="room_306")
    robot.ws_client.sub_imu_data()

    v0 = 0
    s0 = 0

    acc_offset_norm = 9.827884938199292
    acc_list = []  # 长度120
    time_last = None

    init_acc = []
    smooth_acc = []

    while time.time() - time_start <= 10:
        imu_data = robot.ws_client.get_imu_data()

        if imu_data is not None:
            time_now = time.time()
            if time_last is None:
                delta_t = 0
            else:
                delta_t = time_now - time_last
            acc = imu_data.get("linear_acceleration")
            acc_x, acc_y, acc_z = acc.get("x"), acc.get("y"), acc.get("z")
            yaw = quaternion_to_euler_dict(ori=imu_data.get("orientation"))

            acc_norm = np.sqrt(np.square(acc_x) + np.square(acc_y) + np.square(acc_z))

            acc_list.append(acc_norm)
            acc_list = acc_list[-60:]

            if len(acc_list) < 60:
                continue

            acc_smooth = smooth_acc(acc_list)[-1]
            init_acc.append(acc_norm)
            smooth_acc.append(acc_smooth)
            # # acc_y = acc.get("y")
            # # acc_y = acc_y -acc_offset_y
            #
            # acc_y = acc_norm - acc_offset_norm

            v = v0 + acc_smooth * delta_t
            s = s0 + (v0 + v) / 2 * delta_t

            v0 = v
            s0 = s

            # print(f"加速度：{acc_norm}")
            acc_list.append(acc_norm)
            print(f"当前速度：{v}, delta_t: {delta_t} 当前移动距离y：{s} ")
            time_last = time_now
        time.sleep(0.004)

    print(f"平均加速度： {np.mean(acc_list)}")

    import matplotlib.pyplot as plt
    plt.plot(range(len(init_acc)), init_acc - np.mean(acc_list), 'r')
    plt.plot(range(len(smooth_acc)), smooth_acc, 'g')
    plt.show()


if __name__ == '__main__':
    robot = Robot()
    # robot.naive_move_with_imu(goal_angular=90, vel_angular=0.1)
    # robot.naive_move_with_imu(goal_distance=0.2, vel_linear=0.1)

    robot.naive_move_with_QR()
