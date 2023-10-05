class ControlData:
    """
    A data structure for passing control signals to executor
    """

    def __init__(self):
        self.robot_index = None
        # self.omega_left = 0
        # self.omega_right = 0

        self.velocity_x = 0
        self.velocity_y = 0
        self.omega=0

class ControlDataXYOmega:
    """
    A data structure for passing control signals to executor
    """

    def __init__(self):
        self.robot_index = None
        # self.omega_left = 0
        # self.omega_right = 0

        self.velocity_x = 0
        self.velocity_y = 0
        self.omega=0

class SensorData:
    """
    A class for record sensor data
    """

    def __init__(self):
        self.robot_index = None
        self.position = None
        self.orientation = None
        self.linear_velocity = None
        self.angular_velocity = None
        self.occupancy_map = None


class SceneData:
    """
    A class for passing data from scene
    """

    def __init__(self):
        self.observation_list = None
        self.adjacency_list = None
        self.position_list = None
        self.orientation_list = None
