from collections import OrderedDict
import numpy as np
import gym
import torch

from habitat_baselines.utils.common import batch_obs
from habitat_baselines.common.obs_transformers import apply_obs_transforms_batch, get_active_obs_transforms
from habitat_baselines.utils.common import get_num_actions

from hierarchichal_policy import NewHierarchicalPolicy

mappings = {
            'dynamic_goal_to_agent_gps_compass' : 'goal_to_agent_gps_compass',
            'dynamic_obj_start_sensor' : 'obj_start_sensor',
            'dynamic_obj_goal_sensor' : 'obj_goal_sensor',
        }
