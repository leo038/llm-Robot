import numpy as np

MIN_POINT_NUM = 50
DISTANCE_THRESHOLD = 0.05

CAR_RANGE = (-0.6, 0, -0.25, 0.25)




def cal_distance(point, ref_point):
    sum = 0
    for idx, x in enumerate(point):
        sum += np.square(x - ref_point[idx])
    return np.sqrt(sum)


def is_not_car_range(point):
    x, y = point[:2]
    if x >= CAR_RANGE[0] and x <= CAR_RANGE[1] and y >= CAR_RANGE[2] and y <= CAR_RANGE[3]:
        return False
    else:
        return True


def exclude_car_range(point_cloud):
    res = list(map(is_not_car_range, point_cloud))

    return point_cloud[res]


def obstacle_detect(point_cloud_data):
    """
    Point_cloud_data: Nx3 的数组， N是点的个数
    """
    # print(f"shape:{point_cloud_data.shape}")

    if point_cloud_data is None:
        return None

    point_cloud_data = exclude_car_range(point_cloud_data)
    # print(f"shape:{point_cloud_data.shape}")

    distance_x_y = list(map(cal_distance, point_cloud_data[:, :2], [[0, 0]] * len(point_cloud_data)))  ## 计算相对原点的距离

    order = np.array(distance_x_y).argsort()  ## 从小到大排序

    point_cloud_data_sort = point_cloud_data[order]

    # print("排序后：\n")
    # print(point_cloud_data_sort[:5])

    for point in point_cloud_data_sort:
        distance_to_ref = map(cal_distance, point_cloud_data, [point] * len(point_cloud_data))
        num = sum(list(map(lambda x: x <= DISTANCE_THRESHOLD, distance_to_ref)))  ## 找出距离小于阈值的点数
        if num >= MIN_POINT_NUM:
            obstacle_dis = cal_distance(point[:2], [0, 0])
            print(f"最近障碍物坐标：{point}, 障碍物距离：{obstacle_dis}")
            return obstacle_dis

    # print(f"distance xy： {distance_x_y[:10]}")
    #
    # point_cloud_data_with_distance = np.concatenate((point_cloud_data, distance_x_y), axis=1)
    # print("合并后：\n")
    # print(point_cloud_data_with_distance[:5])
    #
    #
    # point_cloud_data_sort = np.sort(point_cloud_data_with_distance, axis=0)
    # print("排序后：\n")
    # print(point_cloud_data_sort[:5])
    #
    #
    # min_idx = np.argmin(distance_x_y)
    # max_idx = np.argmax(distance_x_y)
    #
    # print(len(distance_x_y), min_idx, distance_x_y[min_idx], point_cloud_data[min_idx])
    # print(len(distance_x_y), max_idx, distance_x_y[max_idx], point_cloud_data[max_idx])
    #
    # min_distance_pos = point_cloud_data[min_idx]
    # distance_to_min_pose = list(map(cal_distance, point_cloud_data, [min_distance_pos] * len(point_cloud_data)))
    # # print(distance_to_min_pose)
    #
    # num = sum(list(map(lambda x: x <= DISTANCE_THRESHOLD, distance_to_min_pose)))
    # print(num)
    #
    # np.concatenate()


if __name__ == "__main__":
    data = np.load("point_cloud.npy")
    obstacle_detect(point_cloud_data=data)
