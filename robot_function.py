import time

from arm_utils.arm import ArmControl
from car_utils.robot import Robot


class SmartRobot():

    def __init__(self):
        self.robot = Robot()
        self.arm = ArmControl()

    def move_forward(self, distance=0.5):
        self.robot.naive_move(goal_distance=distance, vel_linear_x=0.2)

    def move_back(self, distance=0.5):
        self.robot.naive_move(goal_distance=distance, vel_linear_x=-0.2)

    def rotate_right(self, angular=90):
        self.robot.naive_move(goal_angular=angular, vel_angular=-0.3)

    def rotate_left(self, angular=90):
        self.robot.naive_move(goal_angular=angular, vel_angular=0.3)

    def gesture_ok(self):
        self.arm.hand_control.gesture_generate(gesture_name="ok")

    def gesture_yeah(self):
        self.arm.hand_control.gesture_generate(gesture_name="yeah")

    def init_arm(self):
        self.arm.init_arm_pose()

    def shake_hands(self):
        self.arm.joints_move(
            joints=[34.92599868774414, 82.20800018310547, 161.96099853515625, 29.17099952697754, 19.94700050354004,
                    64.04399871826172, 47.891998291015625])

    def close_hand(self):
        self.arm.hand_control.gesture_generate(gesture_name="close_hand")

    def open_hand(self):
        self.arm.hand_control.gesture_generate(gesture_name="open_hand")
        res = self.arm.arm.Get_Current_Arm_State()
        print(f"机械臂当前状态： {res}")

    def fight(self, times=1):
        pose1 = [34.27399826049805, 104.98600006103516, 161.90199279785156, 30.197999954223633, 17.20400047302246,
                 92.56900024414062, 47.80500030517578]

        pose2 = [32.34600067138672, 70.93000030517578, 163.68099975585938, 34.303001403808594, 25.209999084472656,
                 45.28499984741211, 47.80099868774414]

        self.arm.hand_control.gesture_generate(gesture_name="close_hand")

        for i in range(times):
            self.arm.joints_move(joints=pose1)
            time.sleep(0.2)

            self.arm.joints_move(joints=pose2)
