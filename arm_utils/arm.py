from .hand import HandControl
from .robotic_arm import *

ARM_VELOCITY = 10  ## 机械臂默认移动速度


class ArmControl():

    def __init__(self):
        self.init_arm()

        self.hand_control = HandControl(arm=self.arm)

    def init_arm(self):
        self.arm = Arm(dev_mode=RM75, ip="192.168.1.18")
        logging.info("设置通讯端口 Modbus RTU 模式")
        self.arm.Set_Modbus_Mode(port=1, baudrate=115200, timeout=2)
        logging.info("打开末端24V电源")
        self.arm.Set_Tool_Voltage(type=3)

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
