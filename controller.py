"""
A controller template
"""
import collections
import math


class ControlData:
    """
    A data structure for passing control signals to executor
    """

    def __init__(self):
        self.robot_index = None
        self.omega_left = 0
        self.omega_right = 0


class Controller:
    """
    A template for robot controller. Support centralized and decentralized control
    """

    def __init__(self):
        """
        desired_distance: Desired formation distance
        centralized_k: For centralized control, a manually defined rate for total velocity
        max_velocity: Maximum linear velocity
        wheel_adjustment: Used for transform linear velocity to angular velocity
        """
        self.desired_distance = 2
        self.centralized_k = 1
        self.max_velocity = 1.2
        self.wheel_adjustment = 10.25
    def velocity_transform(self,velocity_x,velocity_y,theta):
        kk = self.centralized_k
        M11 = kk * math.sin(theta) + math.cos(theta)
        M12 = -kk * math.cos(theta) + math.sin(theta)
        M21 = -kk * math.sin(theta) + math.cos(theta)
        M22 = kk * math.cos(theta) + math.sin(theta)

        wheel_velocity_left = M11 * velocity_x + M12 * velocity_y
        wheel_velocity_right = M21 * velocity_x + M22 * velocity_y

        if (
                math.fabs(wheel_velocity_right) >= math.fabs(wheel_velocity_left)
                and math.fabs(wheel_velocity_right) > self.max_velocity
        ):
            alpha = self.max_velocity / math.fabs(wheel_velocity_right)
        elif (
                math.fabs(wheel_velocity_right) < math.fabs(wheel_velocity_left)
                and math.fabs(wheel_velocity_left) > self.max_velocity
        ):
            alpha = self.max_velocity / math.fabs(wheel_velocity_left)
        else:
            alpha = 1

        wheel_velocity_left = alpha * wheel_velocity_left
        wheel_velocity_right = alpha * wheel_velocity_right
        return wheel_velocity_left,wheel_velocity_right

    def centralized_control(self, index, sensor_data, network_data):
        """
        A centralized control, Expert control
        :param index: Robot index
        :param sensor_data: Data from robot sensor
        :param network_data: Data from the scene
        :return: Control data
        """
        out_put = ControlData()
        if not network_data:
            out_put.omega_left = 0
            out_put.omega_right = 0
            return out_put
        self_robot_index = sensor_data.robot_index
        self_position = sensor_data.position
        self_orientation = sensor_data.orientation
        self_x = self_position[0]
        self_y = self_position[1]
        neighbors = network_data[index]
        velocity_sum_x = 0
        velocity_sum_y = 0
        for neighbor in neighbors:
            rate = (neighbor[3] - self.desired_distance) / neighbor[3]
            velocity_x = rate * (self_x - neighbor[1])
            velocity_y = rate * (self_y - neighbor[2])
            velocity_sum_x -= velocity_x
            velocity_sum_y -= velocity_y
        # transform speed to wheels speed
        theta=sensor_data.orientation[2]
        wheel_velocity_left,wheel_velocity_right=self.velocity_transform(velocity_sum_x,velocity_sum_y,theta)
        out_put.robot_index = self_robot_index
        out_put.omega_left = wheel_velocity_left * self.wheel_adjustment
        out_put.omega_right = wheel_velocity_right * self.wheel_adjustment
        return out_put
    def centralized_control_line(self,index, sensor_data, network_data,separation=2,angle=math.pi/4,max_distance=10,K_c=2):
        """
        A centralized controller to form a line, Expert control
        :param index: Dobot index
        :param sensor_data: Densor_data from robots' sensors or simulators
        :param network_data: Data from the scene
        :param separation: Distance between each robots
        :param angle: Desired line angle to the positive part of x axis
        :param max_distance: Robots only interact within this region
        :return: Control data
        """
        out_put = ControlData()
        if not network_data:
            out_put.omega_left = 0
            out_put.omega_right = 0
            return out_put
        target_position=[]
        robot_num=len(network_data)
        for i in range(robot_num):
            target_position.append([i*separation*math.cos(angle),i*separation*math.sin(angle)])
        robot_position_dict=collections.defaultdict(tuple)
        for key,value in network_data.items():
            for item in value:
                robot_position_dict[item[0]]=(item[1],item[2])
        neighbor_list=[]
        desired_force =[]
        for ri in range(robot_num):
            if not ri==index and ((robot_position_dict[index][0] - robot_position_dict[ri][0]) ** 2 +
                                  (robot_position_dict[index][1] - robot_position_dict[ri][1]) ** 2)**0.5<max_distance:
                neighbor_list.append((robot_position_dict[ri][0],robot_position_dict[ri][1]))
                desired_force.append((target_position[index][0]-target_position[ri][0],
                                      target_position[index][1]-target_position[ri][1]))
        #### get speed
        velocity_index_x=0
        velocity_index_y = 0
        self_position=sensor_data.position[:2]
        theta=sensor_data.orientation[2]
        for i in range(len(neighbor_list)):
            w_ij=[self_position[0]-neighbor_list[i][0]-desired_force[i][0],
                  self_position[1]-neighbor_list[i][1]-desired_force[i][1]]
            velocity_index_x = velocity_index_x-K_c*w_ij[0]
            velocity_index_y = velocity_index_y - K_c * w_ij[1]
        wheel_velocity_left, wheel_velocity_right = self.velocity_transform(velocity_index_x, velocity_index_y, theta)

        self_robot_index = sensor_data.robot_index
        out_put.robot_index = self_robot_index
        out_put.omega_left = wheel_velocity_left * self.wheel_adjustment
        out_put.omega_right = wheel_velocity_right * self.wheel_adjustment
        return out_put
    def decentralized_control(self, sensor_data):
        """
        Decentralized controller
        :param sensor_data:
        :return:
        """
        return sensor_data
