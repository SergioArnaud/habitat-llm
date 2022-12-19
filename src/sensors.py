import numpy as np
from gym import spaces

import habitat
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes
from habitat.tasks.rearrange.rearrange_sensors import MultiObjSensor
from habitat.tasks.utils import cartesian_to_polar

from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
from habitat.tasks.rearrange.utils import (
    UsesRobotInterface,
    batch_transform_point,
)
from habitat.config.default_structured_configs import LabSensorConfig
from dataclasses import dataclass


@registry.register_sensor
class DynamicNavGoalPointGoalSensor(UsesRobotInterface, Sensor):
    """
    GPS and compass sensor relative to the starting object position or goal
    position.
    """

    cls_uuid: str = "dynamic_goal_to_agent_gps_compass"

    def __init__(self, *args, sim, task, **kwargs):
        self._task = task
        self._sim = sim
        super().__init__(*args, task=task, **kwargs)

    def _get_uuid(self, *args, **kwargs):
        return DynamicNavGoalPointGoalSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(
            shape=(2,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, task, *args, **kwargs):
        try:
            target = self._sim.dynamic_target
        except:
            target = np.zeros(3)
        #target = task.nav_goal_pos
        transform = self._sim.get_robot_data(self.robot_id).robot.base_transformation
        dir_vector = transform.inverted().transform_point(target)
        rho, phi = cartesian_to_polar(dir_vector[0], dir_vector[1])
        return np.array([rho, -phi], dtype=np.float32)


@dataclass
class DynamicNavGoalPointGoalSensorConfig(LabSensorConfig):
    type: str = "DynamicNavGoalPointGoalSensor"


@registry.register_sensor
class DynamicTargetStartSensor(UsesRobotInterface, MultiObjSensor):
    """
    Relative position from end effector to target object
    """

    cls_uuid: str = "dynamic_obj_start_sensor"

    def _get_uuid(self, *args, **kwargs):
        return DynamicTargetStartSensor.cls_uuid

    def get_observation(self, *args, observations, episode, **kwargs):
        self._sim: RearrangeSim 

        try:
            target = self._sim.dynamic_target
        except:
            target = np.zeros(3)
        transform = self._sim.get_robot_data(self.robot_id).robot.ee_transform
        dir_vector = transform.inverted().transform_point(target)
        return np.array(dir_vector, dtype=np.float32).reshape(-1)

@dataclass
class DynamicTargetStartSensorConfig(LabSensorConfig):
    type: str = "DynamicTargetStartSensor"
    goal_format: str = "CARTESIAN"
    dimensionality: int = 3


@registry.register_sensor
class DynamicTargetGoalSensor(UsesRobotInterface, MultiObjSensor):
    """
    Relative position from end effector to target object
    """

    cls_uuid: str = "dynamic_obj_goal_sensor"

    def _get_uuid(self, *args, **kwargs):
        return DynamicTargetGoalSensor.cls_uuid

    def get_observation(self, *args, observations, episode, **kwargs):
        self._sim: RearrangeSim 

        try:
            target = self._sim.dynamic_target
        except:
            target = np.zeros(3)
        transform = self._sim.get_robot_data(self.robot_id).robot.ee_transform
        dir_vector = transform.inverted().transform_point(target)
        return np.array(dir_vector, dtype=np.float32).reshape(-1)

@dataclass
class DynamicTargetGoalSensorConfig(LabSensorConfig):
    type: str = "DynamicTargetGoalSensor"
    goal_format: str = "CARTESIAN"
    dimensionality: int = 3


ALL_SENSORS = [ 
    DynamicNavGoalPointGoalSensorConfig,
    DynamicTargetStartSensorConfig,
    DynamicTargetGoalSensorConfig
]
def register_sensors(conf):
    with habitat.config.read_write(conf):
        for sensor_config in ALL_SENSORS:
            SensorConfig = sensor_config()
            conf.habitat.task.lab_sensors[SensorConfig.type] = SensorConfig