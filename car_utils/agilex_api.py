#!/usr/bin/env python
import array
import base64
import copy
import json
import math
import struct
import time
from threading import Timer

import numpy as np
import requests
from websocket import create_connection

try:
    from .task_config import global_logger
except:
    from task_config import global_logger

GLOBAL_STATUS_DICT = {
    0: "等待新任务",
    1: "正在执行任务",
    2: "取消任务",
    3: "任务完成",
    4: "任务中断",
    5: "任务暂停",
    6: "定位丢失",
    7: "异常状态,一般最后一个点无法到达会反馈此状态",
    8: "任务异常暂停，一般中间某个点无法到达会反馈此状态",
    9: "正在充电",
    10: "返回充电桩",
    11: "正在出桩",
    12: "返回起点",
    13: "未出桩",
    14: "等待充电信号"
}

GLOBAL_CONTROL_MODE_DICT = {
    0: "空闲待机状态",
    1: "CAN指令控制",
    2: "串口控制(不支持)",
    3: "手柄遥控控制"
}


def map_coordinate_to_png(pos, map_info):
    '''
        坐标转换函数:将导航时实际使用坐标转换成地图png图片坐标。
        输入参数:地图导航时的实际使用坐标, 地图png绑定的yaml信息数据,包含原点,宽度高度,分辨率。
        输出参数:转成png坐标(x,y)
    '''
    map_pos_x = (pos[0] - map_info['originX']) / map_info['resolution']
    map_pos_y = map_info['gridHeight'] - (pos[1] - map_info['originY']) / map_info['resolution']
    return (map_pos_x, map_pos_y)


def png_coordinate_to_map(pos, map_info):
    """
        坐标转换函数:将地图png图片坐标转换成导航时实际使用坐标。
        输入参数:png坐标(x,y), 地图png绑定的yaml信息数据,包含原点,宽度高度,分辨率。
        输出参数:转成后的实际使用坐标
    """

    png_x = pos[0] * map_info['resolution'] + map_info['originX']

    png_y = (map_info['gridHeight'] - pos[1]) * map_info['resolution'] + map_info['originY'];
    return (png_x, png_y)


def quaternion_from_euler(roll, pitch, yaw):
    """
        角度转换成四元素,返回四元素
    """
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)

    return [qx, qy, qz, qw]


def quaternion_to_euler(ori):
    """
        四元素转成弧度，再转换成角度,返回yaw 角度
    """
    roll = math.atan2(2 * (ori.w * ori.x + ori.y * ori.z),
                      1 - 2 * (ori.x * ori.x + ori.y * ori.y))
    pitch = math.asin(2 * (ori.w * ori.y - ori.x * ori.z))
    yaw = math.atan2(2 * (ori.w * ori.z + ori.x * ori.y),
                     1 - 2 * (ori.z * ori.z + ori.y * ori.y))
    # print('^' * 100, roll, pitch, yaw)
    yaw = math.degrees(yaw)

    # print('&' * 100, roll, pitch, yaw)
    return yaw


def quaternion_to_euler_dict(ori):
    """
        四元素转成弧度，再转换成角度,返回yaw 角度
    """
    roll = math.atan2(2 * (ori.get('w') * ori.get('x') + ori.get('y') * ori.get('z')),
                      1 - 2 * (ori.get('x') * ori.get('x') + ori.get('y') * ori.get('y')))
    pitch = math.asin(2 * (ori.get('w') * ori.get('y') - ori.get('x') * ori.get('z')))
    yaw = math.atan2(2 * (ori.get('w') * ori.get('z') + ori.get('x') * ori.get('y')),
                     1 - 2 * (ori.get('z') * ori.get('z') + ori.get('y') * ori.get('y')))
    yaw = math.degrees(yaw)

    return yaw


class WSClient:
    def __init__(self, address):
        self.address = address
        self.ws = create_connection(address)
        self.isconnect = True
        self.input_data = {
            "op": "call_service",
            "service": "/input/op",
            "type": "tools_msgs/input",
            "args": {
                "file_name": "",
                "op_type": "",
                "id_type": ""
            }
        }

        self.heart_beat_data = None
        self.set_navi_params_data = None
        self.navigation_data = None
        self.slam_status_data = None

        self.localization_status_data = None
        self.robot_pos_data = None
        self.robot_status_data = None

        self.task_status_data = None
        self.task_progress_data = None
        self.global_status_data = None
        self.nav_path_data = None

        self.points_cloud_data = None

        self.battery_data = None

        self.imu_data = None

        ## 自动充电功能设置
        self.set_charge_point_flag = False
        self.charge_point_pose = None
        self.set_charge_level_flag = False

    def update_queue(self, queue, new_item):
        if queue.full():
            queue.get()
            queue.put(new_item)
        else:
            queue.put(new_item)

    def get_queue_item(self, queue):
        res = None
        if not queue.empty():
            res = queue.get()
        return res

    def recieve_data(self):
        if self.isconnect == True:
            try:
                recv_data = json.loads(self.ws.recv())
                # global_logger.info(f"返回数据： {recv_data}")
                if isinstance(recv_data, dict):
                    self.assign_recv_data(recv_data=copy.deepcopy(recv_data))
            except Exception as e:
                res = self.ws.recv()
                # if isinstance(res, bytes):  ##点云数据
                #     data_array = array.array('B', res)
                #     print(f'原始点云大小： {np.array(data_array).shape, data_array[:20], data_array[-20:]}')
                #     self.points_cloud_data = np.array(data_array).reshape(-1, 32)
                #     print(f'原始点云大小： {self.points_cloud_data.shape}')
                global_logger.info(f"获取数据失败。 错误信息： {e}")

            time1 = Timer(0.01, self.recieve_data)  ## 最高数据帧率为50
            time1.start()

    # recv_data.get("topic") == "/run_management/task_status":
    # recv_data.get("topic") == "/run_management/task_status":

    def assign_recv_data(self, recv_data):

        ######################topic#################################################
        # global_logger.info("*" * 50, recv_data)
        if recv_data.get("topic") == "/scan_info":
            # global_logger.info('++' * 100, recv_data.get('msg').get("odom_pose").get('pose').get("position"))
            # self.update_queue(queue=self.robot_pos_queue, new_item=recv_data)
            self.robot_status_data = recv_data

        elif recv_data.get("topic") == "/run_management/task_status":
            # global_logger.info("8" * 100, "task_status:", recv_data)
            # self.update_queue(queue=self.task_status_queue, new_item=recv_data)
            self.task_status_data = recv_data

        elif recv_data.get("topic") == "/run_management/task_progress":
            # global_logger.info("9" * 100, "task_progress:", recv_data)
            self.task_progress_data = recv_data
            # self.update_queue(queue=self.task_progress_queue, new_item=recv_data)

        elif recv_data.get("topic") == "/run_management/global_status":
            # global_logger.info("7" * 100, "global_status:", recv_data)
            self.global_status_data = recv_data
            # global_logger.info(recv_data)

        # topic相关返回结果
        elif recv_data.get("topic") == "/slam_status":
            # self.update_queue(queue=self.slam_status_queue, new_item=recv_data)
            self.slam_status_data = recv_data

        elif recv_data.get("topic") == "/run_management/visual_path":
            # print("*"*100, "路径数据", recv_data)
            self.nav_path_data = recv_data

        elif recv_data.get("topic") == "/localization_lost":
            # self.update_queue(queue=self.localization_status_queue, new_item=recv_data)
            self.localization_status_data = recv_data
        elif recv_data.get("topic") == "/odom":
            # global_logger.info("pose  ："*20,recv_data)
            # self.update_queue(queue=self.robot_pos_queue, new_item=recv_data)
            self.robot_pos_data = recv_data
        # elif recv_data.get("topic") == "/interface_management/robot_status":
        #     self.update_queue(queue=self.robot_status_queue, new_item=recv_data)
        elif recv_data.get("topic") == "/dash_board/robot_status":
            # self.update_queue(queue=self.robot_status_queue, new_item=recv_data)
            self.robot_status_data = recv_data


        elif recv_data.get("topic") == "/points_raw":
            # print("*" * 100)
            # print(recv_data)
            self.points_cloud_data = recv_data


        elif recv_data.get("topic") == "/interface_management/BMS_status":
            self.battery_data = recv_data

        elif recv_data.get("topic") == "/imu/data":
            self.imu_data = recv_data


        ##############################service######################################

        elif recv_data.get("service") == "/run_management/set_navi_params":
            # self.update_queue(queue=self.set_navi_params_queue, new_item=recv_data)
            self.set_navi_params_data = recv_data
        elif recv_data.get("service") == "/input/op":
            # self.update_queue(queue=self.navigation_queue, new_item=recv_data)
            self.navigation_data = recv_data
        elif recv_data.get("service") == "/find_charger/set_charge_point":
            self.set_charge_point_flag = recv_data.get('result')
            global_logger.info(f"设置充电点返回数据：{recv_data}")
        elif recv_data.get("service") == "/find_charger/get_charge_point":
            global_logger.info(f"获取充电点返回数据：{recv_data}")
            if recv_data.get('result'):
                self.charge_point_pose = recv_data.get('values').get("charge_point")
        elif recv_data.get("service") == "/find_charger/set_charge_level":
            global_logger.info(f"设置充电阈值返回数据：{recv_data}")
            if recv_data.get('result'):
                self.set_charge_level_flag = recv_data.get("values").get("success")


        ################################other###########################################

        elif recv_data.get("op") == "pong":
            self.heart_beat_data = recv_data
            # self.update_queue(queue=self.heart_beat_queue, new_item=recv_data)

        else:
            global_logger.info(f"指令返回结果：{recv_data}")

    def check_connect(self):
        connect_status = self.ws.getstatus()
        if connect_status != 101:
            global_logger.warning(f"连接断开， 将重连!!!")
            self.ws = create_connection(self.address)

    def send_msg(self, args, sleep_time=None):
        if self.ws is not None:
            self.check_connect()
            msg = json.dumps(args, ensure_ascii=False).encode("utf-8")
            self.ws.send(msg)
            # if sleep_time is not None:
            #     time.sleep(sleep_time)

            # return json.loads(self.ws.recv())

    def publish_data(self, args):
        if self.ws is not None:
            msg = json.dumps(args, ensure_ascii=False).encode("utf-8")
            self.ws.send(msg)

    def get_bytes_data(self, args):
        if self.ws is not None:
            msg = json.dumps(args, ensure_ascii=False).encode("utf-8")
            self.ws.send(msg)

            return self.ws.recv()

    def heart_beat(self):

        if self.isconnect == True:
            message = {
                "op": "ping",
                "timeStamp": str(time.time() * 1000).split(".")[0]
            }
            self.send_msg(message)
            time1 = Timer(1, self.heart_beat)
            time1.start()

    def on_close(self):
        if self.ws is not None and self.isconnect:
            self.ws.close()
            self.isconnect = False

    def call_input(self, op_type, id_type, file_name=''):
        self.input_data['args']['file_name'] = file_name
        self.input_data['args']['op_type'] = op_type
        self.input_data['args']['id_type'] = id_type

        self.send_msg(self.input_data)

    def record_bag(self, optype, filename=''):
        global_logger.info(f"发送录包指令")
        self.call_input(op_type=optype, id_type="record_data", file_name=filename)

    def mapping_3d(self, optype, filename=''):
        global_logger.info(f"发送3D建图指令")
        self.call_input(op_type=optype, id_type='map_3d', file_name=filename)

    def record_and_mapping_3d(self, optype, filename=''):
        global_logger.info(f"发送同时录包和3D建图指令")
        self.call_input(op_type=optype, id_type='record_and_map', file_name=filename)

    def mapping_2d(self, optype, filename=''):
        global_logger.info(f"发送2D建图指令")
        self.call_input(op_type=optype, id_type='map_2d', file_name=filename)

    def navigation(self, optype, filename=''):
        global_logger.info(f"导航开始指令")
        self.call_input(op_type=optype, id_type='follow_line', file_name=filename)

        time.sleep(0.05)

        # res = self.get_queue_item(queue=self.navigation_queue)
        res = self.navigation_data
        # global_logger.info(f"导航开始指令返回结果： {res}")

    def naive_move(self, vel_linear_x=0, vel_linear_y=0, vel_angular=0):
        msg = {
            "op": "publish",
            "topic": "/cmd_vel",
            "type": "geometry_msgs/Twist",
            "msg": {
                "linear": {
                    "x": vel_linear_x,
                    "y": vel_linear_y,
                    "z": 0
                },
                "angular": {
                    "x": 0,
                    "y": 0,
                    "z": vel_angular
                }
            }
        }
        # print(f"msg:{msg}")
        self.send_msg(msg)

    def change_map(self, filename=''):
        global_logger.info(f"发送切换地图指令， 新地图：{filename}")
        self.call_input(op_type="start", id_type='change_map', file_name=filename)

        time.sleep(0.05)

    def initial_pos(self, pos):
        x, y, theta = pos
        qua = quaternion_from_euler(0, 0, theta * (math.pi / 180) * -1)

        msg = {
            "op": "publish",
            "topic": "/initialpose",
            "type": "geometry_msgs/PoseWithCovarianceStamped",
            "msg": {
                "header": {"frame_id": "map_2d"},
                "pose": {
                    "pose": {
                        "position": {
                            "x": x,
                            "y": y,
                            "z": 0
                        },
                        "orientation": {
                            "w": qua[3],
                            "z": qua[2],
                            "y": qua[1],
                            "x": qua[0]
                        }
                    },
                    "covariance": [
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                    ]
                }
            }
        }
        self.publish_data(msg)

    def set_navi_params(self, max_vel=0.7, max_acc=0.1):
        msg = {
            "op": "call_service",
            "service": "/run_management/set_navi_params",
            "type": "tools_msgs/SetNavisParams",
            "args": {
                "data": [
                    {
                        "name": "navis_ctrl_max_vel_x",
                        "data": str(max_vel)
                    },
                    {
                        "name": "navis_ctrl_max_acc_x",
                        "data": str(max_acc)
                    }
                ],
                "params_type": 0
            }
        }
        global_logger.info(f"发送设置导航参数指令")
        self.send_msg(msg)

    def get_set_navi_result(self):
        # res = self.get_queue_item(queue=self.set_navi_params_queue)
        res = self.set_navi_params_data
        return res
        # global_logger.info(f"发送设置导航参数指令返回结果： {res}")

    def set_charge_level(self, threshold=[20, 90]):
        msg = {
            "op": "call_service",
            "service": "/find_charger/set_charge_level",
            "type": "tools_msgs/setChargeLevel",
            "args": {
                "min_charge_level": threshold[0],
                "max_charge_level": threshold[1]
            }
        }
        global_logger.info(f"发送设置充电阈值指令")
        self.send_msg(msg)

        time.sleep(5)
        global_logger.info(f"设置充电阈是否成功： {self.set_charge_level_flag}")

    def set_charge_point(self):
        msg = {
            "op": "call_service",
            "service": "/find_charger/set_charge_point",
            "type": "std_srvs/SetBool",
            "args": {
            }
        }
        global_logger.info(f"发送设置充电点指令")
        self.send_msg(msg)

        time.sleep(5)
        global_logger.info(f"设置充电点是否成功：{self.set_charge_point_flag}")

    def get_charge_point(self):
        msg = {
            "op": "call_service",
            "service": "/find_charger/get_charge_point",
            "type": "tools_msgs/getChargePoint",
            "args": {}
        }
        global_logger.info(f"发送获取充电点指令")
        self.send_msg(msg)

        time.sleep(5)
        global_logger.info(f"充点电位置: {self.charge_point_pose}")

    def sub_slam_status(self):
        msg = {
            "op": "subscribe",
            "topic": "/slam_status",
            "type": "tools_msgs/slamStatus"
            # "type":"nav_msgs/Odometry"
        }
        global_logger.info(f"发送订阅设备状态指令")
        self.send_msg(msg)

    def get_slam_status(self):
        # res = self.get_queue_item(queue=self.slam_status_queue)
        res = self.slam_status_data
        return res
        # global_logger.info(f"发送订阅设备状态指令返回结果： {res}")

    def sub_nav_status(self):
        msg = {
            "op": "subscribe",
            "topic": "/run_management/global_status",
            "type": "support_ros/GlobalStatus"
            # "type":"nav_msgs/Odometry"
        }
        global_logger.info(f"发送订阅导航状态指令")
        self.send_msg(msg)

    def get_nav_status(self):
        # res = self.get_queue_item(queue=self.nav_status_queue)
        res = self.global_status_data
        return res
        # global_logger.info(f"发送订阅导航状态指令返回结果： {res}")

    def sub_task_status(self):
        msg = {
            "op": "subscribe",
            "topic": "/run_management/task_status",
            "type": "support_ros/TaskStatus"
            # "type":"nav_msgs/Odometry"
        }
        global_logger.info(f"发送订阅任务状态指令")
        self.send_msg(msg)

    def get_task_status(self):
        # res = self.get_queue_item(queue=self.task_status_queue)
        # return res
        return self.task_status_data
        # global_logger.info(f"发送订阅任务状态指令返回结果： {res}")

    def sub_robot_satus(self):
        # msg = {
        #     "op": "subscribe",
        #     "topic": "/interface_management/robot_status",
        #     "type": "tools_msgs/RobotStatus"
        # }

        msg = {
            "op": "subscribe",
            "topic": "/dash_board/robot_status",
            "type": "tools_msgs/RobotStatus"
        }

        global_logger.info(f"发送获取机器人底盘数据指令")
        self.send_msg(msg)

        # time.sleep(0.5)

    def get_robot_status(self):
        # res = self.get_queue_item(queue=self.robot_status_queue)
        res = self.robot_status_data
        return res

    def sub_battery_satus(self):
        msg = {
            "op": "subscribe",
            "topic": "/interface_management/BMS_status",
            "type": "tools_msgs/RobotBmsStatus"
        }

        global_logger.info(f"发送获取机器人电池数据指令")
        self.send_msg(msg)
        # time.sleep(0.5)

    def get_battery_status(self):
        res = self.battery_data.get("msg").get("SOC")
        return res

    def sub_imu_data(self):
        msg = {
            "op": "subscribe",
            "topic": "/imu/data",
            "type": "sensor_msgs/Imu"
        }
        global_logger.info(f"发送获取imu数据指令")
        self.send_msg(msg)

    def unsub_imu_data(self):
        msg = {
            "op": "ubsubscribe",
            "topic": "/imu/data",
            "type": "sensor_msgs/Imu"
        }
        global_logger.info(f"发送取消获取imu数据指令")
        self.send_msg(msg)

    def get_imu_data(self):
        res = None
        if self.imu_data is not None:
            res = self.imu_data.get("msg")

        return res

    def sub_task_progress(self):
        msg = {
            "op": "subscribe",
            "topic": "/run_management/task_progress",
            "type": "std_msgs/Float64"
        }
        global_logger.info(f"发送获取任务进度指令")
        self.send_msg(msg)

    def get_task_progress(self):
        return self.task_progress_data
        # res = self.get_queue_item(queue=self.task_progress_queue)
        # return res

    def cancel_nav(self):
        msg = {
            "op": "publish",
            "topic": "/run_management/navi_task/cancel",
            "type": "actionlib_msgs/GoalID",
            "msg": {
                "stamp": {
                    "secs": 0,
                    "nsecs": 0
                },
                "id": ""
            }
        }
        global_logger.info(f"发送取消导航指令")
        self.publish_data(msg)

    def pause_nav(self):

        msg = {
            "op": "call_service",
            "service": "/run_management/pause",
            "type": "actionlib_msgs/GoalID",
            "args": {
                "pause": True,
                "reason": 0
            }
        }

        global_logger.info(f"发送暂停导航指令")
        self.send_msg(msg)

    def continue_nav(self):
        msg = {
            "op": "call_service",
            "service": "/run_management/pause",
            "type": "actionlib_msgs/GoalID",
            "args": {
                "pause": False,
                "reason": 0
            }
        }

        global_logger.info(f"发送恢复导航指令")
        self.send_msg(msg)

    def sub_pointCloud2(self):
        msg = {
            "op": "subscribe",
            "topic": "/points_raw",
            # "compression": "cbor"  # 如果带有这条字段,获取到的是bytes
        }

        self.send_msg(msg)

        # # compression 字段使用下面接口获取数据
        # res = self.get_bytes_data(msg)
        # # 得到真正的原始数组数据
        # data_array = array.array('B', res)
        #
        # # 无compression 字段使用默认的发送数据接口
        # '''
        #     res = self.send_msg(msg)
        #     data= res.get("msg").get("data")
        #     decode_data = base64.b64decode(data, altchars=None)
        #     # 得到真正的原始数组数据
        #     data_array = array.array('B',decode_data)

        # '''

    def sub_pointCloud2_compress(self):
        # msg = {
        #     "op": "subscribe",
        #     "topic": "/points_raw",
        #     "compression": "cbor"  # 如果带有这条字段,获取到的是bytes
        # }
        msg = {

            "op": "subscribe",
            "topic": "/points_raw",
            "compression": "cbor-point2",
            "type": "sensor_msgs/PointCloud2"
        }

        self.send_msg(msg)

    def get_pointCloud2(self):
        res = None
        if self.points_cloud_data is not None:
            msg = self.points_cloud_data.get("msg")
            init_data = msg.get("data")
            # print(f"3D点云信息：width: {msg.get('width')}, height:{msg.get('height')}, is_bigendian: {msg.get('is_bigendian')},{len(msg.get('fields'))},fields: {msg.get('fields')}, , point_step: {msg.get('point_step')}, row_step: {msg.get('row_step')}, point_num:{msg.get('row_step')/msg.get('point_step')}")
            decode_data = base64.b64decode(init_data, altchars=None)
            point_len = msg.get("point_step")
            point_num = int(msg.get('row_step') / point_len)

            arr_data = array.array('B', decode_data)
            arr_data = np.array(arr_data)
            print(f'原始点云大小： {arr_data.shape}')

            res = arr_data.reshape((point_num, point_len))

        return res

    def get_pointCloud2_compress(self):

        if self.points_cloud_data is not None:

            time_start = time.time()
            msg = self.points_cloud_data.get("msg")
            bounds = msg['_data_uint16']['bounds']
            points_data = base64.b64decode(msg['_data_uint16']['points'])
            points_view = memoryview(points_data)

            points = np.zeros(len(points_data) // 2, dtype=np.float32)

            xrange = bounds[1] - bounds[0]
            xmin = bounds[0]
            yrange = bounds[3] - bounds[2]
            ymin = bounds[2]
            zrange = bounds[5] - bounds[4]
            zmin = bounds[4]

            for i in range(len(points_data) // 6):
                offset = i * 6
                points[3 * i] = (struct.unpack_from('H', points_view, offset)[0] / 65535) * xrange + xmin
                points[3 * i + 1] = (struct.unpack_from('H', points_view, offset + 2)[0] / 65535) * yrange + ymin
                points[3 * i + 2] = (struct.unpack_from('H', points_view, offset + 4)[0] / 65535) * zrange + zmin

            # ## 高效解包方式
            #
            # def _unpack_xyz(xyz_hex):
            #
            # points_data = points_data.reshape(-1,6)
            #

            points = points.reshape(-1, 3)
            print("*" * 100, points.shape)
            print(f"解析点云数据耗时： {time.time() - time_start}")
            print("前5个点：\n")
            print(points[:5])

            return points

    def sub_scan(self):
        msg = {
            "op": "subscribe",
            "topic": "/scan",
            "compression": "cbor"  # 如果带有这条字段,获取到的是bytes，

        }

        # 有compression 字段使用下面接口获取数据
        res = self.get_bytes_data(msg)
        # 得到真正的原始数组数据
        data_array = array.array('B', res)
        global_logger.info(data_array)

        # 无compression 字段使用默认的发送数据接口
        # res = self.send_msg(msg)
        # 得到真正的原始数组数据
        # data= res.get("msg").get("ranges")
        # global_logger.info(data)

    def sub_camera_pointCloud(self):
        msg = {
            "op": "subscribe",
            "topic": "/camera/color/image_raw",
            "compression": "cbor"
        }

        # compression 字段使用下面接口获取数据
        res = self.get_bytes_data(msg)
        data_array = array.array('B', res)

    def sub_robot_pos(self):
        msg = {
            "op": "subscribe",
            "topic": "/odom",
            "type": "nav_msgs/Odometry"
        }

        # msg = {
        #     "op": "subscribe",
        #     "topic": "/scan_info",
        #     "type": "tools_msgs/scanInfo"
        #
        # }

        global_logger.info(f"发送订阅机器人位置指令")
        self.send_msg(msg)
        # res = self.get_queue_item(queue=self.robot_pos_queue)
        # global_logger.info(f"发送订阅机器人位置指令返回结果： {res}")
        # if res is not None:
        #     pos = res.get("msg").get("pose").get("pose")
        # else:
        #     pos = None
        # return pos

    def get_robot_pos(self):
        # res = self.get_queue_item(queue=self.robot_pos_queue)
        res = self.robot_pos_data
        return res

    def sub_plan_path(self):
        msg = {
            "op": "subscribe",
            "topic": "/run_management/visual_path",
            "type": "nav_msgs/Path"
        }

        global_logger.info(f"发送获取导航路线指令")
        self.send_msg(msg)

    def get_nav_path(self):
        res = self.nav_path_data
        return res

    def sub_localiztion_status(self):
        msg = {
            "op": "subscribe",
            "topic": "/localization_lost",
            "type": "std_msgs/Bool"
        }
        global_logger.info(f"发送获取定位状态指令")
        self.send_msg(msg)
        # res = self.get_queue_item(queue=self.localization_status_queue)
        # global_logger.info(f"发送获取定位状态指令返回结果： {res}")
        # if res is not None:
        #     if res.get('msg').get('data'):
        #         global_logger.info(f"定位丢失！！！")
        #     else:
        #         global_logger.info(f"定位准确！！！")

    def get_localization_lost(self):
        # res = self.get_queue_item(queue=self.localization_status_queue)
        # global_logger.info(f"发送获取定位状态指令返回结果： {res}")
        res = self.localization_status_data

        flag = True
        if res is not None:
            flag = res.get('msg').get('data')
        return flag


class HttpClient():
    def __init__(self, url):
        self.url = url
        self.token = None
        self.map_list = None

    def login_(self):
        url = self.url + '/user/passport/login'

        payload = json.dumps({
            "username": "agilex",
            "password": "NTdhMTE5NGNiMDczY2U4YjNiYjM2NWU0YjgwNWE5YWU="
        })
        headers = {
            'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=payload)
        token = json.loads(response.text)
        self.token = token.get('data')

    def get_maplist(self):
        url = self.url + "/map_list?page=1&limit=-1&sortType=&reverse="

        headers = {
            'Authorization': self.token
        }

        response = requests.request("GET", url, headers=headers)
        res_json = json.loads(response.text)
        self.map_list = res_json.get('data')

    def get_map_png(self, map_name):
        url = self.url + "/downloadpng?mapName=" + map_name

        headers = {
            'Authorization': self.token,
            'Content-Type': 'application/x-www-form-urlencoded'
        }

        response = requests.request("GET", url, headers=headers)

        filepath = ('./{}.png'.format(map_name))
        with open(filepath, 'wb') as f:
            # global_logger.info(f"the png content: {response.content}")
            f.write(response.content)

    def get_map_info(self, map_name):
        url = self.url + "/map_info?mapName=" + map_name

        headers = {
            'Authorization': self.token,
            'Content-Type': 'application/json'
        }

        response = requests.request("GET", url, headers=headers)

        res_json = json.loads(response.text)
        if res_json.get('data'):
            return res_json.get('data')['mapInfo']
        else:
            return None

    def run_realtime_task(self, pos: list):

        payload = json.dumps(
            {
                "loopTime": 1,
                "points": [{
                    "position": {
                        "x": pos[0],
                        "y": pos[1],
                        "theta": pos[2]
                    },
                    "isNew": False,
                    "cpx": 0,
                    "cpy": 0
                }],
                "mode": "point"  # mode = path 时需要设置多个点,进行路径导航
            })

        url = self.url + "/realtime_task"

        headers = {
            'Authorization': self.token,
            'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=payload)

        res_json = json.loads(response.text)

        if res_json.get('code') == 0 or res_json.get('successed') == True:
            flag = True
            global_logger.info("下发移动机器人指令成功!!!")
        else:
            flag = False
            global_logger.warning("下发移动机器人指令失败!!!")
        return flag

    def run_list_task(self, map_name, task_name, looptime: int):
        payload = {
            "mapName": map_name,
            "taskName": task_name,
            "loopTime": looptime
        }

        url = self.url + "/run_list_task"

        headers = {
            'Authorization': self.token,
            'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=json.dumps(payload))

        global_logger.info(f"执行列表任务返回结果： {response}")

        res_json = json.loads(response.text)
        if res_json.get('success') == True:
            global_logger.info('执行列表任务指令发送成功！！！')

    def set_list_task(self, task_payload):

        url = self.url + "/set_task"

        headers = {
            'Authorization': self.token,
            'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=json.dumps(task_payload))

        res_json = json.loads(response.text)
        if res_json.get('code') == 0 or res_json.get('successed') == True:
            global_logger.info('设置列表任务指令发送成功！！！')
