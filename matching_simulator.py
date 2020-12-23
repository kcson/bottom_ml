import numpy as np
from util import im2col
import matplotlib.pyplot as plt
import random
import math


class Driver:
    def __init__(self, idx, x, y):
        self.idx = idx
        self.current_x = x
        self.current_y = y
        self.moving_dis = 0
        self.history_x = []
        self.history_y = []
        self.history_x.append(x)
        self.history_y.append(y)
        self.dis = 0
        self.matching_count = 0
        self.dis_score = 0
        self.matching_score = 0
        self.total_score = 0

    def add_matching_count(self):
        self.matching_count += 1

    def add_moving_dis(self):
        self.moving_dis += self.dis

    def calculate_total_score(self, dis_min, matching_min, alpha, beta):
        d_score = (100 / (self.dis + 0.001)) * (dis_min + 0.001)
        m_score = (100 / (self.matching_count + 0.001)) * (matching_min + 0.001)

        self.total_score = d_score * alpha + m_score * beta

    def set_position_history(self, x, y):
        self.history_x.append(x)
        self.history_y.append(y)
        self.current_x = x
        self.current_y = y


is_parking = True
driver_num = 3
matching_count = 60
parking_log_x = 50
parking_log_y = 50

# 거리
alpha = 1
# matching 수
beta = 0

epoch = 1000

epoch_distance = 0
epoch_max_matching = 0
epoch_min_matching = 0
for _ in range(epoch):
    drivers = []
    for i in range(driver_num):
        driver = Driver(i + 1, parking_log_x, parking_log_y)
        drivers.append(driver)

    for j in range(matching_count):
        call_x = random.randint(1, 100)
        call_y = random.randint(1, 100)
        dis_min = 1024 * 1024
        matching_min = 1024 * 1024

        for d in drivers:
            d.dis = math.sqrt((d.current_x - call_x) ** 2 + (d.current_y - call_y) ** 2)
            if d.dis < dis_min:
                dis_min = d.dis
            if d.matching_count < matching_min:
                matching_min = d.matching_count

        for d in drivers:
            d.calculate_total_score(dis_min, matching_min, alpha, beta)

        drivers = sorted(drivers, key=lambda d: d.total_score, reverse=True)
        matching_driver = drivers[0]
        # print(str(matching_driver.idx) + " : " + str(matching_driver.total_score) + " : " + str(matching_driver.matching_count))

        matching_driver.add_matching_count()
        # 주차
        if is_parking:
            # 현위치->호출위치
            matching_driver.add_moving_dis()
            matching_driver.set_position_history(call_x, call_y)
            # 호출위치->주차장
            matching_driver.dis = math.sqrt((matching_driver.current_x - parking_log_x) ** 2 + (matching_driver.current_y - parking_log_y) ** 2)
            matching_driver.add_moving_dis()
            matching_driver.set_position_history(parking_log_x, parking_log_y)
        # 출차
        else:
            if matching_driver.current_x == parking_log_x and matching_driver.current_y == parking_log_y:
                # 현위치->호출위치
                matching_driver.add_moving_dis()
                matching_driver.set_position_history(call_x, call_y)
            else:
                # 현위치 -> 주차장
                matching_driver.dis = math.sqrt((matching_driver.current_x - parking_log_x) ** 2 + (matching_driver.current_y - parking_log_y) ** 2)
                matching_driver.add_moving_dis()
                matching_driver.set_position_history(parking_log_x, parking_log_y)
                # 주차장 -> 호출위치
                matching_driver.dis = math.sqrt((matching_driver.current_x - call_x) ** 2 + (matching_driver.current_y - call_y) ** 2)
                matching_driver.add_moving_dis()
                matching_driver.set_position_history(call_x, call_y)

        is_parking = not is_parking

    total_distance = 0
    for index, driver in enumerate(drivers):
        print(str(driver.idx) + " : " + str(driver.moving_dis) + " : " + str(driver.matching_count))
        total_distance += driver.moving_dis
    epoch_distance += total_distance
    print(total_distance)

print("average : " + str(epoch_distance / epoch))

markers = ['o--', 'v--', 's--']
legend = []
for index, driver in enumerate(drivers):
    legend.append('Linker' + str(index + 1))
    plt.plot(driver.history_x, driver.history_y, markers[index])
    for idx in range(len(driver.history_x)):
        plt.text(driver.history_x[idx] + 0.1, driver.history_y[idx] + 0.1, "{}".format(idx + 1))
plt.legend(legend)
# plt.show()
