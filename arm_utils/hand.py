gesture_dict = {
    "open_hand": [[0, 100, 100, 100, 100, 0], [0, 0, 0, 0, 0, 0]],
    "close_hand": [[0, 100, 100, 100, 100, 0], [90, 100, 100, 100, 100, 100]],
    "hold": [[0, 99, 99, 99, 99, 0], [99, 99, 99, 99, 99, 99]],
    "ok": [[39, 49, 0, 0, 0, 78]],
    "fuck": [[0, 100, 0, 100, 100, 0], [90, 100, 0, 100, 100, 100]],
    "yeah": [[0, 0, 0, 100, 100, 0], [100, 0, 0, 100, 100, 100]]
}


def uint16_to_uint8_pair(A):
    # 提取A的高8位作为B
    B = (A >> 8) & 0xFF

    # 提取A的低8位作为C
    C = A & 0xFF

    return B, C


def hand_percent(data):
    if not isinstance(data, list) or len(data) != 6:
        raise ValueError("the input data must be a list, length is 6")
    control_data = []
    for val in data:
        if val < 0 or val > 100:
            raise ValueError("the data must in range: 0~100")

        B, C = uint16_to_uint8_pair(int(val / 100 * 65535))
        control_data.append(B)
        control_data.append(C)
    return control_data


class HandControl():
    def __init__(self, arm):
        self.arm = arm

    def gesture_generate(self, gesture_name="ok", control_data=None):
        if control_data is not None:
            control_data = hand_percent(control_data)
            flag = self.arm.Write_Registers(port=1, address=1135, num=6, single_data=control_data, device=2, block=True)
            if flag != 0:
                return False
            return True
        gesture_data = gesture_dict.get(gesture_name)
        for data in gesture_data:
            control_data = hand_percent(data)
            flag = self.arm.Write_Registers(port=1, address=1135, num=6, single_data=control_data, device=2, block=True)
            if flag != 0:
                return False
        return True
