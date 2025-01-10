import math

import numpy as np


def smooth_acc(init_acc):
    # # smooth the acc using fft
    assert len(init_acc) == 60
    # acc_diff = np.array(init_acc) - np.mean(init_acc)
    padding_num = 2
    acc_sliding_padding = [0] * padding_num + list(init_acc) + [0] * padding_num

    num_points = len(acc_sliding_padding)
    fhat = np.fft.fft(acc_sliding_padding, num_points)

    window = np.ones(shape=64)

    window[28:36] = 0  ## 去掉高频分量

    fhat = window * fhat
    acc_smooth = np.fft.ifft(fhat)
    acc_smooth = np.real(acc_smooth)

    acc_smooth = acc_smooth[padding_num:num_points - padding_num]

    return acc_smooth


def isRotationMatrix(R):
    Rt = np.transpose(R)  # 旋转矩阵R的转置
    shouldBeIdentity = np.dot(Rt, R)  # R的转置矩阵乘以R
    I = np.identity(3, dtype=R.dtype)  # 3阶单位矩阵
    n = np.linalg.norm(I - shouldBeIdentity)  # np.linalg.norm默认求二范数
    return n < 1e-6  # 目的是判断矩阵R是否正交矩阵（旋转矩阵按道理须为正交矩阵，如此其返回值理论为0）


def rotationMatrixToAngles(R):
    assert (isRotationMatrix(R))  # 判断是否是旋转矩阵（用到正交矩阵特性）

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])  # 矩阵元素下标都从0开始（对应公式中是sqrt(r11*r11+r21*r21)），sy=sqrt(cosβ*cosβ)

    singular = sy < 1e-6  # 判断β是否为正负90°

    if not singular:  # β不是正负90°
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:  # β是正负90°
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)  # 当z=0时，此公式也OK，上面图片中的公式也是OK的
        z = 0

    ## 转成角度
    coff = 180 / 3.1415
    x = x * coff
    y = y * coff
    z = z * coff
    return np.array([x, y, z])


def calculate_vector_angle(vector1, vector2):
    ## 计算2个向量的旋转方向和旋转角度
    x1, y1 = vector1[0]
    x2, y2 = vector1[1]
    v1 = [x2 - x1, y2 - y1]

    x3, y3 = vector2[0]
    x4, y4 = vector2[1]
    v2 = [x4 - x3, y4 - y3]

    # 计算两个向量的点积
    dot_product = np.dot(v1, v2)

    # 计算两个向量的模
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    # 计算两个向量的夹角
    angle = np.arccos(dot_product / (norm_v1 * norm_v2))

    ## 通过2个向量的叉乘计算角度,
    ## ori> 0 向量v2在向量v1的逆时针方向， ori< 0, 向量v2在v1的顺时针方向, ori=0, 共线
    ori = v1[0] * v2[1] - v2[0] * v1[1]

    if ori != 0:
        ori = ori / abs(ori)

    return np.degrees(angle), ori  # 转换为角度


def cal_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return np.sqrt(np.square(x2 - x1) + np.square(y2 - y1))


if __name__ == "__main__":
    vector1 = [[0, 0], [1, 0]]
    vector2 = [[0, 0], [-1, -1]]
    res = calculate_vector_angle(vector1, vector2)
    print(f"夹角： {res}")
